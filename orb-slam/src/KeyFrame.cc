/**
* This file is part of ORB-SLAM.
*
* Copyright (C) 2014 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <http://webdiis.unizar.es/~raulmur/orbslam/>
*
* ORB-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#include "KeyFrame.h"
#include "Converter.h"
#include <ros/ros.h>

namespace ORB_SLAM
{

long unsigned int KeyFrame::nNextId=0;

KeyFrame::KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB):
    //把传入的Frame的一些参数保存下来，并初始化一些KeyFrame特有的参数值
    mnFrameId(F.mnId),  mTimeStamp(F.mTimeStamp), mfGridElementWidthInv(F.mfGridElementWidthInv),
    mfGridElementHeightInv(F.mfGridElementHeightInv), mnTrackReferenceForFrame(0),mnBALocalForKF(0), mnBAFixedForKF(0),
    mnLoopQuery(0), mnRelocQuery(0),fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy),//
    mBowVec(F.mBowVec),//
    im(F.im), mnMinX(F.mnMinX), mnMinY(F.mnMinY), mnMaxX(F.mnMaxX), mnMaxY(F.mnMaxY), mK(F.mK),//
    mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn),//
    mDescriptors(F.mDescriptors.clone()),//复制Frame的描述子
    mvpMapPoints(F.mvpMapPoints),//
    mpKeyFrameDB(pKFDB),//
    mpORBvocabulary(F.mpORBvocabulary),//复制vocabulary
    mFeatVec(F.mFeatVec),//
    mbFirstConnection(true), mpParent(NULL), mbNotErase(false), mbToBeErased(false), mbBad(false),
    mnScaleLevels(F.mnScaleLevels), mvScaleFactors(F.mvScaleFactors), mvLevelSigma2(F.mvLevelSigma2),
    mvInvLevelSigma2(F.mvInvLevelSigma2), mpMap(pMap)//
{
    mnId=nNextId++;

    mnGridCols=FRAME_GRID_COLS;
    mnGridRows=FRAME_GRID_ROWS;
    mGrid.resize(mnGridCols);
    //将Frame网格记录的关键点索引值复制过来
    for(int i=0; i<mnGridCols;i++)
    {
        mGrid[i].resize(mnGridRows);
        for(int j=0; j<mnGridRows; j++)
            mGrid[i][j] = F.mGrid[i][j];
    }
    //把Frame的位姿赋值过来
    SetPose(F.mTcw);    
}

void KeyFrame::ComputeBoW()
{
    if(mBowVec.empty() || mFeatVec.empty())
    {
        //将从Frame复制过来的描述子转化为vector<cv::Mat>
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        // Feature vector associate features with nodes in the 4th level (from leaves up)
        // We assume the vocabulary tree has 6 levels, change the 4 otherwise
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

void KeyFrame::SetPose(const cv::Mat &Rcw,const cv::Mat &tcw)
{
    boost::mutex::scoped_lock lock(mMutexPose);
    Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
    tcw.copyTo(Tcw.col(3).rowRange(0,3));

    Ow=-Rcw.t()*tcw;
}

void KeyFrame::SetPose(const cv::Mat &Tcw_)
{
    boost::mutex::scoped_lock lock(mMutexPose);
    Tcw_.copyTo(Tcw);
    cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = Tcw.rowRange(0,3).col(3);
    Ow = -Rcw.t()*tcw;
}

cv::Mat KeyFrame::GetPose()
{
    boost::mutex::scoped_lock lock(mMutexPose);
    return Tcw.clone();
}

cv::Mat KeyFrame::GetPoseInverse()
{
    boost::mutex::scoped_lock lock(mMutexPose);
    cv::Mat Twc = cv::Mat::eye(4,4,Tcw.type());
    cv::Mat Rwc = (Tcw.rowRange(0,3).colRange(0,3)).t();
    cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);
    Rwc.copyTo(Twc.rowRange(0,3).colRange(0,3));
    twc.copyTo(Twc.rowRange(0,3).col(3));
    return Twc.clone();
}

cv::Mat KeyFrame::GetProjectionMatrix()
{
    boost::mutex::scoped_lock lock(mMutexPose);
    return mK*Tcw.rowRange(0,3);
}

cv::Mat KeyFrame::GetCameraCenter()
{
    boost::mutex::scoped_lock lock(mMutexPose);
    return Ow.clone();
}

cv::Mat KeyFrame::GetRotation()
{
    boost::mutex::scoped_lock lock(mMutexPose);
    return Tcw.rowRange(0,3).colRange(0,3).clone();
}

cv::Mat KeyFrame::GetTranslation()
{
    boost::mutex::scoped_lock lock(mMutexPose);
    return Tcw.rowRange(0,3).col(3).clone();
}

void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)
{
    {
        boost::mutex::scoped_lock lock(mMutexConnections);
        if(!mConnectedKeyFrameWeights.count(pKF))
            mConnectedKeyFrameWeights[pKF]=weight;
        else if(mConnectedKeyFrameWeights[pKF]!=weight)
            mConnectedKeyFrameWeights[pKF]=weight;
        else
            return;
    }

    UpdateBestCovisibles();
}

void KeyFrame::UpdateBestCovisibles()
{
    boost::mutex::scoped_lock lock(mMutexConnections);
    vector<pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());
    for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
       vPairs.push_back(make_pair(mit->second,mit->first));
    //从小到大排序
    sort(vPairs.begin(),vPairs.end());
    list<KeyFrame*> lKFs;
    list<int> lWs;
    for(size_t i=0, iend=vPairs.size(); i<iend;i++)
    {
        //大的在前，小的在后
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }
    //更新数据
    mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
    mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());    
}

set<KeyFrame*> KeyFrame::GetConnectedKeyFrames()
{
    boost::mutex::scoped_lock lock(mMutexConnections);
    set<KeyFrame*> s;
    for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin();mit!=mConnectedKeyFrameWeights.end();mit++)
        s.insert(mit->first);
    return s;
}

vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames()
{
    boost::mutex::scoped_lock lock(mMutexConnections);
    return mvpOrderedConnectedKeyFrames;
}

vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
{
    boost::mutex::scoped_lock lock(mMutexConnections);
    //如果mvpOrderedConnectedKeyFrames中没有N个关键帧，直接全给出
    if((int)mvpOrderedConnectedKeyFrames.size()<N)
        return mvpOrderedConnectedKeyFrames;
    else
        //要是多于N个，则给出前N个
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(),mvpOrderedConnectedKeyFrames.begin()+N);

}

vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(const int &w)
{
    boost::mutex::scoped_lock lock(mMutexConnections);

    if(mvpOrderedConnectedKeyFrames.empty())
        return vector<KeyFrame*>();

    vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(),mvOrderedWeights.end(),w,KeyFrame::weightComp);
    if(it==mvOrderedWeights.end())
        return vector<KeyFrame*>();
    else
    {
        int n = it-mvOrderedWeights.begin();
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin()+n);
    }
}

int KeyFrame::GetWeight(KeyFrame *pKF)
{
    boost::mutex::scoped_lock lock(mMutexConnections);
    if(mConnectedKeyFrameWeights.count(pKF))
        return mConnectedKeyFrameWeights[pKF];
    else
        return 0;
}

void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx)
{
    boost::mutex::scoped_lock lock(mMutexFeatures);
    //这里是以关键点索引idx为key，MapPoint为value
    //即可以看作是形成了一个多对一的map
    //这里的idx是mvpMapPoints向量的索引值，并不是先前任务的是Map结构vector<MapPoint*> mvpMapPoints
    //此idx对应的是当前关键帧对应的关键点的索引值 -2018-12-15
    mvpMapPoints[idx]=pMP;
}

void KeyFrame::EraseMapPointMatch(const size_t &idx)
{
    boost::mutex::scoped_lock lock(mMutexFeatures);
    mvpMapPoints[idx]=NULL;
}

void KeyFrame::EraseMapPointMatch(MapPoint* pMP)
{
    int idx = pMP->GetIndexInKeyFrame(this);
    if(idx>=0)
        mvpMapPoints[idx]=NULL;
}


void KeyFrame::ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP)
{
    mvpMapPoints[idx]=pMP;
}

set<MapPoint*> KeyFrame::GetMapPoints()
{
    boost::mutex::scoped_lock lock(mMutexFeatures);
    set<MapPoint*> s;
    for(size_t i=0, iend=mvpMapPoints.size(); i<iend; i++)
    {
        if(!mvpMapPoints[i])
            continue;
        MapPoint* pMP = mvpMapPoints[i];
        if(!pMP->isBad())
            s.insert(pMP);
    }
    return s;
}

int KeyFrame::TrackedMapPoints()
{
    boost::mutex::scoped_lock lock(mMutexFeatures);

    int nPoints=0;
    for(size_t i=0, iend=mvpMapPoints.size(); i<iend; i++)
    {
        if(mvpMapPoints[i])
            nPoints++;
    }

    return nPoints;
}

vector<MapPoint*> KeyFrame::GetMapPointMatches()
{
    boost::mutex::scoped_lock lock(mMutexFeatures);
    return mvpMapPoints;
}

MapPoint* KeyFrame::GetMapPoint(const size_t &idx)
{
    boost::mutex::scoped_lock lock(mMutexFeatures);
    return mvpMapPoints[idx];
}

cv::KeyPoint KeyFrame::GetKeyPointUn(const size_t &idx) const
{
    return mvKeysUn[idx];
}

int KeyFrame::GetKeyPointScaleLevel(const size_t &idx) const
{
    return mvKeysUn[idx].octave;
}

cv::Mat KeyFrame::GetDescriptor(const size_t &idx)
{
    return mDescriptors.row(idx).clone();
}

cv::Mat KeyFrame::GetDescriptors()
{
    return mDescriptors.clone();
}

vector<cv::KeyPoint> KeyFrame::GetKeyPoints() const
{
    return mvKeys;
}

vector<cv::KeyPoint> KeyFrame::GetKeyPointsUn() const
{
    return mvKeysUn;
}

cv::Mat KeyFrame::GetCalibrationMatrix() const
{
    return mK.clone();
}

DBoW2::FeatureVector KeyFrame::GetFeatureVector()
{
    boost::mutex::scoped_lock lock(mMutexFeatures);
    return mFeatVec;
}

DBoW2::BowVector KeyFrame::GetBowVector()
{
    boost::mutex::scoped_lock lock(mMutexFeatures);
    return mBowVec;
}

cv::Mat KeyFrame::GetImage()
{
    boost::mutex::scoped_lock lock(mMutexImage);
    return im.clone();
}
//1.统计当前关键帧与其他关键帧的共视数量
//2.将当前关键帧添加到共视数量满足阈值设定的关键帧中，如果都不满足阈值设定，则添加到共视最多的关键帧中
//3.当前关键帧记录所有与其共视的关键帧，并分别从大到小记录关键帧和共视数量
//4.取与当前关键帧共视数量最多的作为当前帧的父节点，并将当前帧添加到父节点的子节点数据结构中
void KeyFrame::UpdateConnections()
{
    map<KeyFrame*,int> KFcounter;

    vector<MapPoint*> vpMP;

    {
        boost::mutex::scoped_lock lockMPs(mMutexFeatures);
        //该关键帧上所有已经匹配的MapPoint
        vpMP = mvpMapPoints;
    }

    //For all map points in keyframe check in which other keyframes are they seen
    //Increase counter for those keyframes
    //统计当前关键帧所有匹配点MapPoint与其他关键帧的共视数量
    for(vector<MapPoint*>::iterator vit=vpMP.begin(), vend=vpMP.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;

        if(!pMP)
            continue;

        if(pMP->isBad())
            continue;
        // 对于每一个MapPoint点，observations记录了可以观测到该MapPoint的所有关键帧
        map<KeyFrame*,size_t> observations = pMP->GetObservations();

        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            //跳过当前帧
            if(mit->first->mnId==mnId)
                continue;
            //统计当前帧与其他关键帧有多少共视的MapPoint
            KFcounter[mit->first]++;
        }
    }

    if(KFcounter.empty())
        return;

    //If the counter is greater than threshold add connection
    //In case no keyframe counter is over threshold add the one with maximum counter
    int nmax=0;
    KeyFrame* pKFmax=NULL;
    int th = 15;

    vector<pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(KFcounter.size());
    //将当前关键帧添加到共视数量满足阈值设定的关键帧中
    for(map<KeyFrame*,int>::iterator mit=KFcounter.begin(), mend=KFcounter.end(); mit!=mend; mit++)
    {
        if(mit->second>nmax)
        {
            //记录计数最大值和对应的关键帧
            nmax=mit->second;
            pKFmax=mit->first;
        }
        //大于阈值的都暂存到vPairs中，并AddConnection
        //这里边可能包括也可能不包括共识最多的那一个，当最大的都没有大于阈值时就不包括
        if(mit->second>=th)
        {
            vPairs.push_back(make_pair(mit->second,mit->first));
            //共识超过阈值，把当前帧添加到与其共识的关键帧mConnectedKeyFrameWeights中
            //即关键帧指针和共识数量
            (mit->first)->AddConnection(this,mit->second);
        }
    }
    //如果没有大于指定阈值的，那么就把当前帧添加到共视计数最大的那一帧中
    if(vPairs.empty())
    {
        vPairs.push_back(make_pair(nmax,pKFmax));
        pKFmax->AddConnection(this,nmax);
    }
    //从小到大排列
    sort(vPairs.begin(),vPairs.end());
    list<KeyFrame*> lKFs;
    list<int> lWs;
    //把vPairs拆分存放
    for(size_t i=0; i<vPairs.size();i++)
    {
        //从头方向加入链表，即最后最大的在最前头，最小的在最后
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    {
        boost::mutex::scoped_lock lockCon(mMutexConnections);

        // mspConnectedKeyFrames = spConnectedKeyFrames;
        //当前关键帧记录了所有与自己共视的关键帧，而并不一定都被其他关键帧也记录
        //即此结构并不是完全双向的，因为是否添加到其他关键帧共视中需要满足前面的阈值设定
        mConnectedKeyFrameWeights = KFcounter;
        //共识数量从大到小
        mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());

        if(mbFirstConnection && mnId!=0)
        {
            //指定与当前帧共视最多的关键帧为其父节点
            mpParent = mvpOrderedConnectedKeyFrames.front();
            //并将当前帧添加到父节点的子节点数据结构中
            mpParent->AddChild(this);
            mbFirstConnection = false;
        }

    }
}

void KeyFrame::AddChild(KeyFrame *pKF)
{
    boost::mutex::scoped_lock lockCon(mMutexConnections);
    mspChildrens.insert(pKF);
}

void KeyFrame::EraseChild(KeyFrame *pKF)
{
    boost::mutex::scoped_lock lockCon(mMutexConnections);
    mspChildrens.erase(pKF);
}


void KeyFrame::ChangeParent(KeyFrame *pKF)
{
    boost::mutex::scoped_lock lockCon(mMutexConnections);
    mpParent = pKF;
    pKF->AddChild(this);
}

set<KeyFrame*> KeyFrame::GetChilds()
{
    boost::mutex::scoped_lock lockCon(mMutexConnections);
    return mspChildrens;
}

KeyFrame* KeyFrame::GetParent()
{
    boost::mutex::scoped_lock lockCon(mMutexConnections);
    return mpParent;
}

bool KeyFrame::hasChild(KeyFrame *pKF)
{
    boost::mutex::scoped_lock lockCon(mMutexConnections);
    return mspChildrens.count(pKF);
}

void KeyFrame::AddLoopEdge(KeyFrame *pKF)
{
    boost::mutex::scoped_lock lockCon(mMutexConnections);
    mbNotErase = true;
    mspLoopEdges.insert(pKF);
}

set<KeyFrame*> KeyFrame::GetLoopEdges()
{
    boost::mutex::scoped_lock lockCon(mMutexConnections);
    return mspLoopEdges;
}

void KeyFrame::SetNotErase()
{
    boost::mutex::scoped_lock lock(mMutexConnections);
    mbNotErase = true;
}

void KeyFrame::SetErase()
{
    {
        boost::mutex::scoped_lock lock(mMutexConnections);
        if(mspLoopEdges.empty())
        {
            mbNotErase = false;
        }
    }

    if(mbToBeErased)
    {
        SetBadFlag();
    }
}

//1.从共视到该关键帧的帧中删除记录，清楚该关键帧观测对应的MapPoint中的记录
//2.为Children分配Parent
//3.从它的Parent中删除，从Map中删除，从关键帧数据库中删除
void KeyFrame::SetBadFlag()
{   
    {
        boost::mutex::scoped_lock lock(mMutexConnections);
        if(mnId==0)
            return;
        else if(mbNotErase)
        {
            mbToBeErased = true;
            return;
        }
    }
    //将当前帧从它的共视帧中删除，并更新共视帧的共视关系
    for(map<KeyFrame*,int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
        //共视帧删除此关键帧，并更新共视关系
        mit->first->EraseConnection(this);
    //清除MapPoint中对该帧的记录
    for(size_t i=0; i<mvpMapPoints.size(); i++)
        if(mvpMapPoints[i])
            mvpMapPoints[i]->EraseObservation(this);
    {
        boost::mutex::scoped_lock lock(mMutexConnections);
        boost::mutex::scoped_lock lock1(mMutexFeatures);

        mConnectedKeyFrameWeights.clear();
        mvpOrderedConnectedKeyFrames.clear();

        // Update Spanning Tree
        set<KeyFrame*> sParentCandidates;
        sParentCandidates.insert(mpParent);

        // Assign at each iteration one children with a parent (the pair with highest covisibility weight)
        // Include that children as new parent candidate for the rest
        //直到把所有的Children都分配完
        while(!mspChildrens.empty())
        {
            bool bContinue = false;

            int max = -1;
            KeyFrame* pC;
            KeyFrame* pP;
            //要被删除的帧的所有的Children
            //1.取出一个Children的共视帧，寻找共视帧和候选Parent中相同的帧，如果有
            //  选择权重最大的那一个作为Parent，插入到Parent的子中，并设置自身的Parent指针
            for(set<KeyFrame*>::iterator sit=mspChildrens.begin(), send=mspChildrens.end(); sit!=send; sit++)
            {
                //取出一个Children
                KeyFrame* pKF = *sit;
                if(pKF->isBad())
                    continue;

                // Check if a parent candidate is connected to the keyframe
                //取出该Children的共视帧
                vector<KeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();
                for(size_t i=0, iend=vpConnected.size(); i<iend; i++)
                {
                    for(set<KeyFrame*>::iterator spcit=sParentCandidates.begin(), spcend=sParentCandidates.end(); spcit!=spcend; spcit++)
                    {
                        //比较共视帧与当前帧所有候选Parent是否存在相同的帧
                        if(vpConnected[i]->mnId == (*spcit)->mnId)
                        {
                            //取出相同帧的权重
                            int w = pKF->GetWeight(vpConnected[i]);
                            //记录下相同帧中权重最大的
                            if(w>max)
                            {
                                pC = pKF;
                                pP = vpConnected[i];
                                max = w;
                                bContinue = true;
                            }
                        }
                    }
                }
            }
            //只要找到了新的有最大权重的相同帧，就继续while循环，否则退出
            if(bContinue)
            {
                pC->ChangeParent(pP);
                sParentCandidates.insert(pC);
                mspChildrens.erase(pC);
            }
            else
                break;
        }

        // If a children has no covisibility links with any parent candidate, assign to the original parent of this KF
        //如果还有没有分配的，那么就把当前帧的Parent作为它们的Parent
        if(!mspChildrens.empty())
            for(set<KeyFrame*>::iterator sit=mspChildrens.begin(); sit!=mspChildrens.end(); sit++)
            {
                (*sit)->ChangeParent(mpParent);
            }
        //删除当前帧
        mpParent->EraseChild(this);
        mbBad = true;
    }

    //从地图中删除当前帧
    mpMap->EraseKeyFrame(this);
    //从关键帧数据库中删除当前帧
    mpKeyFrameDB->erase(this);
}

bool KeyFrame::isBad()
{
    boost::mutex::scoped_lock lock(mMutexConnections);
    return mbBad;
}
//删除某个共视帧，并更新当前帧的共视关系
void KeyFrame::EraseConnection(KeyFrame* pKF)
{
    bool bUpdate = false;
    {
        boost::mutex::scoped_lock lock(mMutexConnections);
        //将pKF从共视帧变量mConnectedKeyFrameWeights中删除
        if(mConnectedKeyFrameWeights.count(pKF))
        {
            mConnectedKeyFrameWeights.erase(pKF);
            bUpdate=true;
        }
    }
    //如果删除了pKF这个共视帧，更新当前帧的另外两个共视帧变量
    //mvpOrderedConnectedKeyFrames和mvOrderedWeights
    if(bUpdate)
        UpdateBestCovisibles();
}

vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r) const
{
    vector<size_t> vIndices;
    vIndices.reserve(mvKeysUn.size());
    //检查边界
    int nMinCellX = floor((x-mnMinX-r)*mfGridElementWidthInv);
    nMinCellX = max(0,nMinCellX);
    if(nMinCellX>=mnGridCols)
        return vIndices;

    int nMaxCellX = ceil((x-mnMinX+r)*mfGridElementWidthInv);
    nMaxCellX = min(mnGridCols-1,nMaxCellX);
    if(nMaxCellX<0)
        return vIndices;

    int nMinCellY = floor((y-mnMinY-r)*mfGridElementHeightInv);
    nMinCellY = max(0,nMinCellY);
    if(nMinCellY>=mnGridRows)
        return vIndices;

    int nMaxCellY = ceil((y-mnMinY+r)*mfGridElementHeightInv);
    nMaxCellY = min(mnGridRows-1,nMaxCellY);
    if(nMaxCellY<0)
        return vIndices;
    //提取关键点
    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            vector<size_t> vCell = mGrid[ix][iy];
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                if(abs(kpUn.pt.x-x)<=r && abs(kpUn.pt.y-y)<=r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool KeyFrame::IsInImage(const float &x, const float &y) const
{
    return (x>=mnMinX && x<mnMaxX && y>=mnMinY && y<mnMaxY);
}
//计算出当前帧所有MapPoint在相机坐标系下的z的坐标值
//并返回1/q处的深度值
float KeyFrame::ComputeSceneMedianDepth(int q)
{
    vector<MapPoint*> vpMapPoints;
    cv::Mat Tcw_;
    {
    boost::mutex::scoped_lock lock(mMutexFeatures);
    boost::mutex::scoped_lock lock2(mMutexPose);
    vpMapPoints = mvpMapPoints;
    Tcw_ = Tcw.clone();
    }

    vector<float> vDepths;
    vDepths.reserve(mvpMapPoints.size());
    //取出仅有z坐标变换有关的R矩阵中的向量
    cv::Mat Rcw2 = Tcw_.row(2).colRange(0,3);
    Rcw2 = Rcw2.t();
    //取出z坐标的平移
    float zcw = Tcw_.at<float>(2,3);
    //计算出当前帧所有MapPoint在相机坐标系下的z的坐标值，放入vDepths中
    for(size_t i=0; i<mvpMapPoints.size(); i++)
    {
        if(mvpMapPoints[i])
        {
            MapPoint* pMP = mvpMapPoints[i];
            cv::Mat x3Dw = pMP->GetWorldPos();
            //计算出MapPoint在相机坐标系下的z坐标
            float z = Rcw2.dot(x3Dw)+zcw;
            vDepths.push_back(z);
        }
    }

    sort(vDepths.begin(),vDepths.end());
    //返回中间值
    return vDepths[(vDepths.size()-1)/q];
}

} //namespace ORB_SLAM
