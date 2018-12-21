/**
* This file is part of ORB-SLAM.
* It is based on the file orb.cpp from the OpenCV library (see BSD license below)
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

#include "ORBmatcher.h"

#include<limits.h>

#include<ros/ros.h>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include<stdint-gcc.h>


using namespace std;

namespace ORB_SLAM
{

const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 50;
const int ORBmatcher::HISTO_LENGTH = 30;


ORBmatcher::ORBmatcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri)
{
}

int ORBmatcher::SearchByProjection(Frame &F, const vector<MapPoint*> &vpMapPoints, const float th)
{
    //记录成功匹配的数量
    int nmatches=0;

    const bool bFactor = th!=1.0;
    //将vpMapPoints中所有可能在当前帧匹配到的都窗搜索匹配一下
    for(size_t iMP=0; iMP<vpMapPoints.size(); iMP++)
    {
        MapPoint* pMP = vpMapPoints[iMP];
        //判断上一步是否认为此点是可Tracking的，可以为true
        if(!pMP->mbTrackInView)
            continue;

        if(pMP->isBad())
            continue;

        const int &nPredictedLevel = pMP->mnTrackScaleLevel;    

        // The size of the window will depend on the viewing direction
        float r = RadiusByViewingCos(pMP->mTrackViewCos);
        //上下两行代码来调整搜索半径
        if(bFactor)
            r*=th;
        //取出可能匹配区域中的当前帧的关键点
        vector<size_t> vNearIndices =
                F.GetFeaturesInArea(pMP->mTrackProjX,pMP->mTrackProjY,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel);

        if(vNearIndices.empty())
            continue;

        cv::Mat MPdescriptor = pMP->GetDescriptor();

        int bestDist=INT_MAX;
        int bestLevel= -1;
        int bestDist2=INT_MAX;
        int bestLevel2 = -1;
        int bestIdx =-1 ;

        // Get best and second matches with near keypoints
        for(vector<size_t>::iterator vit=vNearIndices.begin(), vend=vNearIndices.end(); vit!=vend; vit++)
        {
            size_t idx = *vit;
            //如果当前帧关键点对应的MapPoint处已经有匹配了，pass
            if(F.mvpMapPoints[idx])
                continue;

            cv::Mat d=F.mDescriptors.row(idx);

            const int dist = DescriptorDistance(MPdescriptor,d);
            //保留最佳匹配和次优匹配距离及level，记录最佳匹配的index
            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestLevel2 = bestLevel;
                bestLevel = F.mvKeysUn[idx].octave;
                bestIdx=idx;
            }
            else if(dist<bestDist2)
            {
                bestLevel2 = F.mvKeysUn[idx].octave;
                bestDist2=dist;
            }
        }

        // Apply ratio to second match (only if best and second are in the same scale level)
        if(bestDist<=TH_HIGH)
        {   //当前两个最佳匹配level相同且距离差异不大时，pass
            if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                continue;
            //把匹配上的MapPoint指针记录到当前帧中
            F.mvpMapPoints[bestIdx]=pMP;
            nmatches++;
        }
    }

    return nmatches;
}

float ORBmatcher::RadiusByViewingCos(const float &viewCos)
{
    if(viewCos>0.998)
        return 2.5;
    else
        return 4.0;
}


bool ORBmatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1,const cv::KeyPoint &kp2,const cv::Mat &F12,const KeyFrame* pKF2)
{
    // Epipolar line in second image l = x1'F12 = [a b c]
    const float a = kp1.pt.x*F12.at<float>(0,0)+kp1.pt.y*F12.at<float>(1,0)+F12.at<float>(2,0);
    const float b = kp1.pt.x*F12.at<float>(0,1)+kp1.pt.y*F12.at<float>(1,1)+F12.at<float>(2,1);
    const float c = kp1.pt.x*F12.at<float>(0,2)+kp1.pt.y*F12.at<float>(1,2)+F12.at<float>(2,2);

    const float num = a*kp2.pt.x+b*kp2.pt.y+c;

    const float den = a*a+b*b;

    if(den==0)
        return false;

    const float dsqr = num*num/den;

    return dsqr<3.84*pKF2->GetSigma2(kp2.octave);
}

int ORBmatcher::SearchByBoW(KeyFrame* pKF,Frame &F, vector<MapPoint*> &vpMapPointMatches)
{
    vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();

    vpMapPointMatches = vector<MapPoint*>(F.mvpMapPoints.size(),static_cast<MapPoint*>(NULL));

    DBoW2::FeatureVector vFeatVecKF = pKF->GetFeatureVector();

    int nmatches=0;

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
    DBoW2::FeatureVector::iterator KFit = vFeatVecKF.begin();
    DBoW2::FeatureVector::iterator Fit = F.mFeatVec.begin();
    DBoW2::FeatureVector::iterator KFend = vFeatVecKF.end();
    DBoW2::FeatureVector::iterator Fend = F.mFeatVec.end();

    while(KFit != KFend && Fit != Fend)
    {
        if(KFit->first == Fit->first)
        {
            vector<unsigned int> vIndicesKF = KFit->second;
            vector<unsigned int> vIndicesF = Fit->second;

            for(size_t iKF=0, iendKF=vIndicesKF.size(); iKF<iendKF; iKF++)
            {
                const unsigned int realIdxKF = vIndicesKF[iKF];

                MapPoint* pMP = vpMapPointsKF[realIdxKF];

                if(!pMP)
                    continue;

                if(pMP->isBad())
                    continue;

                cv::Mat dKF= pKF->GetDescriptor(realIdxKF);

                int bestDist1=INT_MAX;
                int bestIdxF =-1 ;
                int bestDist2=INT_MAX;

                for(size_t iF=0, iendF=vIndicesF.size(); iF<iendF; iF++)
                {
                    const unsigned int realIdxF = vIndicesF[iF];

                    if(vpMapPointMatches[realIdxF])
                        continue;

                    cv::Mat dF = F.mDescriptors.row(realIdxF).clone();

                    const int dist =  DescriptorDistance(dKF,dF);

                    if(dist<bestDist1)
                    {
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdxF=realIdxF;
                    }
                    else if(dist<bestDist2)
                    {
                        bestDist2=dist;
                    }
                }

                if(bestDist1<=TH_LOW)
                {
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                    {
                        vpMapPointMatches[bestIdxF]=pMP;

                        cv::KeyPoint kp = pKF->GetKeyPointUn(realIdxKF);

                        if(mbCheckOrientation)
                        {
                            float rot = kp.angle-F.mvKeys[bestIdxF].angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            ROS_ASSERT(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdxF);
                        }
                        nmatches++;
                    }
                }

            }

            KFit++;
            Fit++;
        }
        else if(KFit->first < Fit->first)
        {
            KFit = vFeatVecKF.lower_bound(Fit->first);
        }
        else
        {
            Fit = F.mFeatVec.lower_bound(KFit->first);
        }
    }


    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vpMapPointMatches[rotHist[i][j]]=NULL;
                nmatches--;
            }
        }
    }

    return nmatches;
}
//把vpPoints通过Sim3投影到当前帧寻找匹配的MapPoint
int ORBmatcher::SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const vector<MapPoint*> &vpPoints, vector<MapPoint*> &vpMatched, int th)
{
    // Get Calibration Parameters for later projection
    const float fx = pKF->fx;
    const float fy = pKF->fy;
    const float cx = pKF->cx;
    const float cy = pKF->cy;

    const int nMaxLevel = pKF->GetScaleLevels()-1;
    vector<float> vfScaleFactors = pKF->GetScaleFactors();

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw/scw;
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
    cv::Mat Ow = -Rcw.t()*tcw;

    // Set of MapPoints already found in the KeyFrame
    set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(NULL);

    int nmatches=0;

    // For each Candidate MapPoint Project and Match
    for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0)
            continue;

        // Project into Image
        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        // Depth must be inside the scale invariance region of the point
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist = cv::norm(PO);

        if(dist<minDistance || dist>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist)
            continue;

        // Compute predicted scale level
        const float ratio = dist/minDistance;

        vector<float>::iterator it = lower_bound(vfScaleFactors.begin(), vfScaleFactors.end(), ratio);
        const int nPredictedLevel = min(static_cast<int>(it-vfScaleFactors.begin()),nMaxLevel);

        // Search in a radius
        const float radius = th*pKF->GetScaleFactor(nPredictedLevel);

        vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            size_t idx = *vit;
            if(vpMatched[idx])
                continue;

            const int kpLevel= pKF->GetKeyPointScaleLevel(idx);

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            cv::Mat dKF = pKF->GetDescriptor(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }

        }

        if(bestDist<=TH_LOW)
        {
            vpMatched[bestIdx]=pMP;
            nmatches++;
        }

    }

    return nmatches;
}
//在以F1的MapPoint对应的关键点(u,v)为中心的区域内
//为F1的已匹配的MapPoint对应的关键点寻找在F2中的最佳匹配点
//并将匹配上的点也更新到F2的mvpMapPoints中
int ORBmatcher::WindowSearch(Frame &F1, Frame &F2, int windowSize, vector<MapPoint *> &vpMapPointMatches2, int minScaleLevel, int maxScaleLevel)
{
    int nmatches=0;
    //先把vector全部置为`NULL`
    //当前Frame的mvpMapPoints是和mvKeys相同大小的vector，初始值为NULL
    //`NULL`表示没有匹配项，其他值表示匹配项的MapPoint的地址指针
    vpMapPointMatches2 = vector<MapPoint*>(F2.mvpMapPoints.size(),static_cast<MapPoint*>(NULL));
    //`-1`表示没有找打匹配项，其他值表示匹配项的索引值index
    vector<int> vnMatches21 = vector<int>(F2.mvKeysUn.size(),-1);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;
    //判断是否有最大level和最小level的限制
    const bool bMinLevel = minScaleLevel>0;
    const bool bMaxLevel= maxScaleLevel<INT_MAX;
    //遍历上一帧所有的已经匹配到的MapPoint
    //此for循环的作用是：
    //1.在以F1.mvpMapPoints的关键点像素坐标为中心，windowSize为半径的区域内寻找在F2中的最佳匹配点
    //2.放入到角度直方图对应的bin中
    //3.
    for(size_t i1=0, iend1=F1.mvpMapPoints.size(); i1<iend1; i1++)
    {
        MapPoint* pMP1 = F1.mvpMapPoints[i1];

        if(!pMP1)
            continue;
        if(pMP1->isBad())
            continue;
        //取出此MapPoint对应的非畸变关键点keypoint
        const cv::KeyPoint &kp1 = F1.mvKeysUn[i1];
        //该关键点对应的level
        int level1 = kp1.octave;
        //检查是否满足level要求
        if(bMinLevel)
            if(level1<minScaleLevel)
                continue;

        if(bMaxLevel)
            if(level1>maxScaleLevel)
                continue;
        //找到在以x,y为中心,边长为2r的区域内且在[minLevel, maxLevel]的当前帧F2的特征点
        vector<size_t> vIndices2 = F2.GetFeaturesInArea(kp1.pt.x,kp1.pt.y, windowSize, level1, level1);

        if(vIndices2.empty())
            continue;

        cv::Mat d1 = F1.mDescriptors.row(i1);

        int bestDist = INT_MAX;
        int bestDist2 = INT_MAX;
        int bestIdx2 = -1;
        //在vIndices2中找到与F1帧某特征点最近的两个
        for(vector<size_t>::iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
        {
            size_t i2 = *vit;
            //如果F2的某个关键点已经分配了MapPoint，则跳过它
            if(vpMapPointMatches2[i2])
                continue;

            cv::Mat d2 = F2.mDescriptors.row(i2);

            int dist = DescriptorDistance(d1,d2);

            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestIdx2=i2;
            } else if(dist<bestDist2)
            {
                bestDist2=dist;
            }
        }
        //判断最匹配的特征点是否足够好
        if(bestDist<=bestDist2*mfNNratio && bestDist<=TH_HIGH)
        {
            //为F2记录此MapPoint地址指针
            vpMapPointMatches2[bestIdx2]=pMP1;
            //为F2记录对应的F1中的关键点索引值
            vnMatches21[bestIdx2]=i1;
            nmatches++;
            //放入对应的bin中
            float rot = F1.mvKeysUn[i1].angle-F2.mvKeysUn[bestIdx2].angle;
            if(rot<0.0)
                rot+=360.0f;
            int bin = round(rot*factor);
            if(bin==HISTO_LENGTH)
                bin=0;
            ROS_ASSERT(bin>=0 && bin<HISTO_LENGTH);
            rotHist[bin].push_back(bestIdx2);
        }
    }
    //如果匹配是正确的，那么绝大多数的点都应该有相同的angle偏转，即都处在同一个bin中
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;
        //找到前三个最大的bin
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i!=ind1 && i!=ind2 && i!=ind3)
            {
                //剔除其他bin中的点
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    //把其他bin中记录的索引值对应的MapPoint地址值都清除
                    vpMapPointMatches2[rotHist[i][j]]=NULL;
                    //索引值置`-1`
                    vnMatches21[rotHist[i][j]]=-1;
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

//此函数使用优化后的相机位姿，为F1中的还未在F2中找到对应匹配点的MapPoint再次寻找匹配点
int ORBmatcher::SearchByProjection(Frame &F1, Frame &F2, int windowSize, vector<MapPoint *> &vpMapPointMatches2)
{
    vpMapPointMatches2 = F2.mvpMapPoints;
    //复制F2中的mvpMapPoints向量
    set<MapPoint*> spMapPointsAlreadyFound(vpMapPointMatches2.begin(),vpMapPointMatches2.end());

    int nmatches = 0;
    //取出刚刚使用g2o优化后的相机位姿
    const cv::Mat Rc2w = F2.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tc2w = F2.mTcw.rowRange(0,3).col(3);

    for(size_t i1=0, iend1=F1.mvpMapPoints.size(); i1<iend1; i1++)
    {
        //取前一帧MapPoint
        MapPoint* pMP1 = F1.mvpMapPoints[i1];
        //如果对应的MapPoint指针为NULL，则跳过
        if(!pMP1)
            continue;
        //如果F1中的此MapPoint已经在F2中有记录了，则跳过
        if(pMP1->isBad() || spMapPointsAlreadyFound.count(pMP1))
            continue;
        //即上述两个判断的作用是：只为F1中还没有在F2中匹配的MapPoint寻找匹配点
        cv::KeyPoint kp1 = F1.mvKeysUn[i1];
        int level1 = kp1.octave;
        //将前一帧的关键点重投影到当前帧
        cv::Mat x3Dw = pMP1->GetWorldPos();
        cv::Mat x3Dc2 = Rc2w*x3Dw+tc2w;

        const float xc2 = x3Dc2.at<float>(0);
        const float yc2 = x3Dc2.at<float>(1);
        const float invzc2 = 1.0/x3Dc2.at<float>(2);
        //在当前帧的重投影像素坐标
        float u2 = F2.fx*xc2*invzc2+F2.cx;
        float v2 = F2.fy*yc2*invzc2+F2.cy;
        //再次窗内搜索
        vector<size_t> vIndices2 = F2.GetFeaturesInArea(u2,v2, windowSize, level1, level1);

        if(vIndices2.empty())
            continue;

        cv::Mat d1 = F1.mDescriptors.row(i1);

        int bestDist = INT_MAX;
        int bestDist2 = INT_MAX;
        int bestIdx2 = -1;

        //寻找最匹配的两个
        for(vector<size_t>::iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
        {
            size_t i2 = *vit;

            if(vpMapPointMatches2[i2])
                continue;

            cv::Mat d2 = F2.mDescriptors.row(i2);

            int dist = DescriptorDistance(d1,d2);

            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestIdx2=i2;
            } else if(dist<bestDist2)
            {
                bestDist2=dist;
            }
        }
        //判断最匹配的点质量是否足够好
        if(static_cast<float>(bestDist)<=static_cast<float>(bestDist2)*mfNNratio && bestDist<=TH_HIGH)
        {
            vpMapPointMatches2[bestIdx2]=pMP1;
            nmatches++;
        }

    }

    return nmatches;
}


//根据第一帧的所有关键点，在第二帧中找到最匹配的店
//然后更新vbPrevMatched，记录新的第二帧中被匹配的点坐标
//vnMatches12记录对应的索引值
int ORBmatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize)
{
    int nmatches=0;
    //记录前一帧所有关键点匹配到当前帧的索引
    vnMatches12 = vector<int>(F1.mvKeysUn.size(),-1);
    //关键点方向bin
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;
    //记录匹配点距离
    vector<int> vMatchedDistance(F2.mvKeysUn.size(),INT_MAX);
    //记录当前帧的特征点匹配到前一帧的索引
    vector<int> vnMatches21(F2.mvKeysUn.size(),-1);
    //遍历第一帧检测到的所有特征点
    for(size_t i1=0, iend1=F1.mvKeysUn.size(); i1<iend1; i1++)
    {
        cv::KeyPoint kp1 = F1.mvKeysUn[i1];
        int level1 = kp1.octave;
        //只使用level=0上提取的关键点
        if(level1>0)
            continue;
        //获取第二帧图像中某一区域范围内的所有特征点的index
        //以前一帧(其实是第一帧)的所有关键点坐标为原点，在半径为windowSize的区域中选取第二针的关键点
        vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x,vbPrevMatched[i1].y, windowSize,level1,level1);

        if(vIndices2.empty())
            continue;
        //取第一帧某一特征点的描述子
        cv::Mat d1 = F1.mDescriptors.row(i1);

        int bestDist = INT_MAX;
        int bestDist2 = INT_MAX;
        int bestIdx2 = -1;
        //遍历计算第二帧中所有可能特征点与第一帧的某个特征点的距离，保留最近的前两个
        for(vector<size_t>::iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
        {
            size_t i2 = *vit;

            cv::Mat d2 = F2.mDescriptors.row(i2);

            int dist = DescriptorDistance(d1,d2);

            if(vMatchedDistance[i2]<=dist)
                continue;

            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestIdx2=i2;
            }
            else if(dist<bestDist2)
            {
                bestDist2=dist;
            }
        }
        //判断最近的是否满足阈值要求
        if(bestDist<=TH_LOW)
        {
            //判断最近的和此近的之间的距离是否足够远，如果它俩太近就舍弃
            if(bestDist<(float)bestDist2*mfNNratio)
            {
                //如果此点被匹配过，那么说明此点不可靠，删除
                if(vnMatches21[bestIdx2]>=0)
                {
                    vnMatches12[vnMatches21[bestIdx2]]=-1;
                    nmatches--;
                }
                vnMatches12[i1]=bestIdx2;
                vnMatches21[bestIdx2]=i1;
                vMatchedDistance[bestIdx2]=bestDist;
                nmatches++;
                //将匹配点放入对应的bin中
                if(mbCheckOrientation)
                {
                    float rot = F1.mvKeysUn[i1].angle-F2.mvKeysUn[bestIdx2].angle;
                    if(rot<0.0)
                        rot+=360.0f;
                    int bin = round(rot*factor);
                    if(bin==HISTO_LENGTH)
                        bin=0;
                    ROS_ASSERT(bin>=0 && bin<HISTO_LENGTH);
                    rotHist[bin].push_back(i1);
                }
            }
        }

    }
    //找出直方图中前三的bin
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;
        //找出最大的前三组bin
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            //跳过匹配数最多的三组bin
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            //把其他bin中记录的匹配特征点全部删除，并递减计数nmatches的值
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                int idx1 = rotHist[i][j];
                if(vnMatches12[idx1]>=0)
                {
                    vnMatches12[idx1]=-1;
                    nmatches--;
                }
            }
        }

    }

    //Update prev matched
    //将当前帧中匹配的点坐标记录到vbPrevMatched前(vnMatches12.size())中，即覆盖了原来的数据
    for(size_t i1=0, iend1=vnMatches12.size(); i1<iend1; i1++)
        if(vnMatches12[i1]>=0)
            vbPrevMatched[i1]=F2.mvKeysUn[vnMatches12[i1]].pt;

    return nmatches;
}
//1.通过BoW缩小匹配范围，然后就是在范围内找最优匹配
//2.剔除不可靠的关键点
//此函数只查找这两帧已经发现的MapPoint之间的最佳匹配，并不产生新的
int ORBmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12)
{
    vector<cv::KeyPoint> vKeysUn1 = pKF1->GetKeyPointsUn();
    DBoW2::FeatureVector vFeatVec1 = pKF1->GetFeatureVector();
    vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    cv::Mat Descriptors1 = pKF1->GetDescriptors();

    vector<cv::KeyPoint> vKeysUn2 = pKF2->GetKeyPointsUn();
    DBoW2::FeatureVector vFeatVec2 = pKF2->GetFeatureVector();
    vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    cv::Mat Descriptors2 = pKF2->GetDescriptors();

    vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(),static_cast<MapPoint*>(NULL));
    vector<bool> vbMatched2(vpMapPoints2.size(),false);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f/HISTO_LENGTH;

    int nmatches = 0;
    //节点id，关键点index
    //FeatureVector-->map<NodeId, std::vector<unsigned int> >
    //BowVector-->std::map<WordId, WordValue>
    DBoW2::FeatureVector::iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::iterator f2end = vFeatVec2.end();
    //此while循环配合最外层的if语句实现查找相同键值key的功能
    //即查找相同节点id，NodeId
    //不同节点id肯定没有公共单词，即匹配
    while(f1it != f1end && f2it != f2end)
        {
            if(f1it->first == f2it->first)
            {
                //遍历f1节点中的所有关键点
                //这两个for实现的功能是：
                //找到在两个相同节点中记录的两帧的关键点间的最佳匹配
                //其实是f1的点暴力匹配f2中的每个点，在f2中找到最匹配的
                for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
                {
                    size_t idx1 = f1it->second[i1];
                    //取出f1节点中的一个MapPoint
                    MapPoint* pMP1 = vpMapPoints1[idx1];
                    if(!pMP1)
                        continue;
                    if(pMP1->isBad())
                        continue;
                    //取出关键点的描述子
                    cv::Mat d1 = Descriptors1.row(idx1);

                    int bestDist1=INT_MAX;
                    int bestIdx2 =-1 ;
                    int bestDist2=INT_MAX;
                    //遍历f2中的所有关键点
                    for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                    {
                        size_t idx2 = f2it->second[i2];
                        //取出f2节点中的一个MapPoint
                        MapPoint* pMP2 = vpMapPoints2[idx2];

                        if(vbMatched2[idx2] || !pMP2)
                            continue;

                        if(pMP2->isBad())
                            continue;
                        //取出关键点的描述子
                        cv::Mat d2 = Descriptors2.row(idx2);
                        //计算距离
                        int dist = DescriptorDistance(d1,d2);
                        //记录最短距离和次短距离
                        if(dist<bestDist1)
                        {
                            bestDist2=bestDist1;
                            bestDist1=dist;
                            bestIdx2=idx2;
                        }
                        else if(dist<bestDist2)
                        {
                            bestDist2=dist;
                        }
                    }
                    //阈值判定
                    if(bestDist1<TH_LOW)
                    {
                        //最短和次短是否相距足够远
                        if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                        {
                            vpMatches12[idx1]=vpMapPoints2[bestIdx2];
                            vbMatched2[bestIdx2]=true;
                            //放入bin中
                            if(mbCheckOrientation)
                            {
                                float rot = vKeysUn1[idx1].angle-vKeysUn2[bestIdx2].angle;
                                if(rot<0.0)
                                    rot+=360.0f;
                                int bin = round(rot*factor);
                                if(bin==HISTO_LENGTH)
                                    bin=0;
                                ROS_ASSERT(bin>=0 && bin<HISTO_LENGTH);
                                rotHist[bin].push_back(idx1);
                            }
                            nmatches++;
                        }
                    }
                }

                f1it++;
                f2it++;
            }
            else if(f1it->first < f2it->first)
            {
                //标记了一个不小于value
                f1it = vFeatVec1.lower_bound(f2it->first);
            }
            else
            {
                f2it = vFeatVec2.lower_bound(f1it->first);
            }
        }
    //用旋转一致性剔除不可靠的关键点
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vpMatches12[rotHist[i][j]]=NULL;
                //vnMatches12[rotHist[i][j]]=-1;
                nmatches--;
            }
        }

    }
    //返回匹配数
    return nmatches;
}
//1.通过DBoW2加速匹配两关键帧中未匹配的MapPoint
//2.将最后匹配的关键点分别放入vMatchedKeys1和vMatchedKeys2中，并用vMatchedPairs记录对应的index
int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
vector<cv::KeyPoint> &vMatchedKeys1, vector<cv::KeyPoint> &vMatchedKeys2, vector<pair<size_t, size_t> > &vMatchedPairs)
{
    vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    vector<cv::KeyPoint> vKeysUn1 = pKF1->GetKeyPointsUn();
    cv::Mat Descriptors1 = pKF1->GetDescriptors();
    DBoW2::FeatureVector vFeatVec1 = pKF1->GetFeatureVector();

    vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    vector<cv::KeyPoint> vKeysUn2 = pKF2->GetKeyPointsUn();
    cv::Mat Descriptors2 = pKF2->GetDescriptors();
    DBoW2::FeatureVector vFeatVec2 = pKF2->GetFeatureVector();

    // Find matches between not tracked keypoints
    // Matching speeded-up by ORB Vocabulary
    // Compare only ORB that share the same node

    int nmatches=0;
    vector<bool> vbMatched2(vKeysUn2.size(),false);
    vector<int> vMatches12(vKeysUn1.size(),-1);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f/HISTO_LENGTH;
    //FeatVec数据结构：map<NodeId, std::vector<unsigned int> >
    DBoW2::FeatureVector::iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::iterator f2end = vFeatVec2.end();
    //仅当至少有一个遍历完才结束循环
    //次循环暂时还看不懂，对DBoW2不了解，跳过，总之是在通过BDoW2加速找到匹配点 -2018-12-14
    while(f1it!=f1end && f2it!=f2end)
    {
        if(f1it->first == f2it->first)
        {
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
                size_t idx1 = f1it->second[i1];

                MapPoint* pMP1 = vpMapPoints1[idx1];

                // If there is already a MapPoint skip
                if(pMP1)
                    continue;

                const cv::KeyPoint &kp1 = vKeysUn1[idx1];

                cv::Mat d1 = Descriptors1.row(idx1);

                vector<pair<int,size_t> > vDistIndex;

                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    size_t idx2 = f2it->second[i2];

                    MapPoint* pMP2 = vpMapPoints2[idx2];

                    // If we have already matched or there is a MapPoint skip
                    if(vbMatched2[idx2] || pMP2)
                        continue;

                    cv::Mat d2 = Descriptors2.row(idx2);

                    const int dist = DescriptorDistance(d1,d2);

                    if(dist>TH_LOW)
                        continue;

                    vDistIndex.push_back(make_pair(dist,idx2));
                }

                if(vDistIndex.empty())
                    continue;

                sort(vDistIndex.begin(),vDistIndex.end());
                int BestDist = vDistIndex.front().first;
                int DistTh = round(2*BestDist);

                for(size_t id=0; id<vDistIndex.size(); id++)
                {
                    if(vDistIndex[id].first>DistTh)
                        break;

                    int currentIdx2 = vDistIndex[id].second;
                    cv::KeyPoint &kp2 = vKeysUn2[currentIdx2];
                    //对极约束检查
                    if(CheckDistEpipolarLine(kp1,kp2,F12,pKF2))
                    {
                        //vbMatched2标识pKF2对应某个索引值的被匹配了
                        vbMatched2[currentIdx2]=true;
                        //vMatches12记录pKF2中被匹配的index
                        vMatches12[idx1]=currentIdx2;
                        nmatches++;

                        if(mbCheckOrientation)
                        {
                            float rot = kp1.angle-kp2.angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            ROS_ASSERT(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(idx1);
                        }

                        break;
                    }

                }
            }

            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }
    //根据旋转一致性剔除一些点
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vMatches12[rotHist[i][j]]=-1;
                nmatches--;
            }
        }

    }

    vMatchedKeys1.clear();
    vMatchedKeys1.reserve(nmatches);
    vMatchedKeys2.clear();
    vMatchedKeys2.reserve(nmatches);
    vMatchedPairs.clear();
    vMatchedPairs.reserve(nmatches);

    for(size_t i=0, iend=vMatches12.size(); i<iend; i++)
    {
        if(vMatches12[i]<0)
            continue;
        //记录pKF1中匹配的关键点
        vMatchedKeys1.push_back(vKeysUn1[i]);
        //记录pKF2中匹配的关键点
        vMatchedKeys2.push_back(vKeysUn2[vMatches12[i]]);
        //记录这两针对应的关键点的index
        vMatchedPairs.push_back(make_pair(i,vMatches12[i]));
    }

    return nmatches;
}
//1.将vpMapPoints重投影到pKF上，寻找匹配点
int ORBmatcher::Fuse(KeyFrame *pKF, vector<MapPoint *> &vpMapPoints, float th)
{
    cv::Mat Rcw = pKF->GetRotation();
    cv::Mat tcw = pKF->GetTranslation();

    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    const int nMaxLevel = pKF->GetScaleLevels()-1;
    vector<float> vfScaleFactors = pKF->GetScaleFactors();

    cv::Mat Ow = pKF->GetCameraCenter();

    int nFused=0;

    for(size_t i=0; i<vpMapPoints.size(); i++)
    {
        MapPoint* pMP = vpMapPoints[i];

        if(!pMP)
            continue;

        if(pMP->isBad() || pMP->IsInKeyFrame(pKF))
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        //MapPoint在当前帧坐标系下的坐标
        cv::Mat p3Dc = Rcw*p3Dw + tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0f)
            continue;
        //计算像素坐标
        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist3D = cv::norm(PO);

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();
        //判断视角是否满足要求
        if(PO.dot(Pn)<0.5*dist3D)
            continue;

        // Compute predicted scale level
        const float ratio = dist3D/minDistance;

        vector<float>::iterator it = lower_bound(vfScaleFactors.begin(), vfScaleFactors.end(), ratio);
        const int nPredictedLevel = min(static_cast<int>(it-vfScaleFactors.begin()),nMaxLevel);

        // Search in a radius
        const float radius = th*vfScaleFactors[nPredictedLevel];
        //取可能匹配的点，对level没有限制
        vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        //取MapPoint的描述子
        cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;
            const int kpLevel= pKF->GetKeyPointScaleLevel(idx);

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            cv::Mat dKF = pKF->GetDescriptor(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist<=TH_LOW)
        {
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            //如果此关键点已经匹配MapPoint
            if(pMPinKF)
            {
                //如果已匹配到的MapPoint并不是bad状态，就Replace
                if(!pMPinKF->isBad()) //令人费解的操作！！！！
                    //传入参数为新产生的MapPoint
                    pMP->Replace(pMPinKF);                
            }
            //如果没匹配则直接添加
            else
            {
                pMP->AddObservation(pKF,bestIdx);
                pKF->AddMapPoint(pMP,bestIdx);
            }
            nFused++;
        }
    }

    return nFused;
}
//通过修正后的位姿将闭环MapPoint重投影到当前帧产生新的MapPoint匹配
int ORBmatcher::Fuse(KeyFrame *pKF, cv::Mat Scw, const vector<MapPoint *> &vpPoints, float th)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw/scw;
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
    cv::Mat Ow = -Rcw.t()*tcw;

    // Set of MapPoints already found in the KeyFrame
    set<MapPoint*> spAlreadyFound = pKF->GetMapPoints();

    const int nMaxLevel = pKF->GetScaleLevels()-1;
    vector<float> vfScaleFactors = pKF->GetScaleFactors();

    int nFused=0;

    // For each candidate MapPoint project and match
    for(size_t iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0f)
            continue;

        // Project into Image
        const float invz = 1.0/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        // Depth must be inside the scale pyramid of the image
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist3D = cv::norm(PO);

        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist3D)
            continue;

        // Compute predicted scale level
        const float ratio = dist3D/minDistance;

        vector<float>::iterator it = lower_bound(vfScaleFactors.begin(), vfScaleFactors.end(), ratio);
        const int nPredictedLevel = min(static_cast<int>(it-vfScaleFactors.begin()),nMaxLevel);

        // Search in a radius of 2.5*sigma(ScaleLevel)
        const float radius = th*pKF->GetScaleFactor(nPredictedLevel);

        vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius

        cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::iterator vit=vIndices.begin(); vit!=vIndices.end(); vit++)
        {
            const size_t idx = *vit;
            const int kpLevel = pKF->GetKeyPointScaleLevel(idx);

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            cv::Mat dKF = pKF->GetDescriptor(idx);

            int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist<=TH_LOW)
        {
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if(pMPinKF)
            {
                if(!pMPinKF->isBad())
                    pMPinKF->Replace(pMP);
            }
            else
            {
                pMP->AddObservation(pKF,bestIdx);
                pKF->AddMapPoint(pMP,bestIdx);
            }
            nFused++;
        }

    }

    return nFused;

}
//此函数其实就是利用已经获得的sim3在两个关键帧之间寻找匹配点
//这里使用了互相验证的方法，鲁棒性更好
int ORBmatcher::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint*> &vpMatches12,
                                   const float &s12, const cv::Mat &R12, const cv::Mat &t12, float th)
{
    const float fx = pKF1->fx;
    const float fy = pKF1->fy;
    const float cx = pKF1->cx;
    const float cy = pKF1->cy;

    // Camera 1 from world
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();

    //Camera 2 from world
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    //Transformation between cameras
    cv::Mat sR12 = s12*R12;
    cv::Mat sR21 = (1.0/s12)*R12.t();
    cv::Mat t21 = -sR21*t12;

    const int nMaxLevel1 = pKF1->GetScaleLevels()-1;
    vector<float> vfScaleFactors1 = pKF1->GetScaleFactors();

    vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const int N1 = vpMapPoints1.size();

    const int nMaxLevel2 = pKF2->GetScaleLevels()-1;
    vector<float> vfScaleFactors2 = pKF2->GetScaleFactors();

    vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const int N2 = vpMapPoints2.size();

    vector<bool> vbAlreadyMatched1(N1,false);
    vector<bool> vbAlreadyMatched2(N2,false);

    for(int i=0; i<N1; i++)
    {
        MapPoint* pMP = vpMatches12[i];
        if(pMP)
        {
            vbAlreadyMatched1[i]=true;
            int idx2 = pMP->GetIndexInKeyFrame(pKF2);
            if(idx2>=0 && idx2<N2)
                vbAlreadyMatched2[idx2]=true;
        }
    }

    vector<int> vnMatch1(N1,-1);
    vector<int> vnMatch2(N2,-1);

    // Transform from KF1 to KF2 and search
    //在KF2中给KF1中的MapPoint找还没在KF2种匹配到的MapPoint
    //使用sim3来进行投影映射：KF1的MapPoint-->KF2像素坐标
    for(int i1=0; i1<N1; i1++)
    {
        MapPoint* pMP = vpMapPoints1[i1];
        //找pKF1中的MapPoint还未和pKF2中的MapPoint匹配的
        if(!pMP || vbAlreadyMatched1[i1])
            continue;

        if(pMP->isBad())
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        //转换到c1相机坐标系
        cv::Mat p3Dc1 = R1w*p3Dw + t1w;
        //转换到c2相机坐标系
        cv::Mat p3Dc2 = sR21*p3Dc1 + t21;

        // Depth must be positive
        if(p3Dc2.at<float>(2)<0.0)
            continue;
        //转换到像素坐标值    
        float invz = 1.0/p3Dc2.at<float>(2);
        float x = p3Dc2.at<float>(0)*invz;
        float y = p3Dc2.at<float>(1)*invz;

        float u = fx*x+cx;
        float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF2->IsInImage(u,v))
            continue;

        float maxDistance = pMP->GetMaxDistanceInvariance();
        float minDistance = pMP->GetMinDistanceInvariance();
        float dist3D = cv::norm(p3Dc2);

        // Depth must be inside the scale invariance region
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        // Compute predicted octave
        float ratio = dist3D/minDistance;

        vector<float>::iterator it = lower_bound(vfScaleFactors2.begin(), vfScaleFactors2.end(), ratio);
        const int nPredictedLevel = min(static_cast<int>(it-vfScaleFactors2.begin()),nMaxLevel2);

        // Search in a radius
        float radius = th*vfScaleFactors2[nPredictedLevel];
        //提取关键点
        vector<size_t> vIndices = pKF2->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        //
        for(vector<size_t>::iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            size_t idx = *vit;

            cv::KeyPoint kp = pKF2->GetKeyPointUn(idx);
            //检查level
            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            cv::Mat dKF = pKF2->GetDescriptor(idx);

            int dist = DescriptorDistance(dMP,dKF);
            //记录最短距离的描述子index
            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch1[i1]=bestIdx;
        }
    }

    // Transform from KF2 to KF2 and search
    for(int i2=0; i2<N2; i2++)
    {
        MapPoint* pMP = vpMapPoints2[i2];
        //找pKF2中的MapPoint还未和pKF1中的MapPoint匹配的
        if(!pMP || vbAlreadyMatched2[i2])
            continue;

        if(pMP->isBad())
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc2 = R2w*p3Dw + t2w;
        //转换到c1坐标系
        cv::Mat p3Dc1 = sR12*p3Dc2 + t12;

        // Depth must be positive
        if(p3Dc1.at<float>(2)<0.0)
            continue;
        //投影到像素KF1像素坐标
        float invz = 1.0/p3Dc1.at<float>(2);
        float x = p3Dc1.at<float>(0)*invz;
        float y = p3Dc1.at<float>(1)*invz;

        float u = fx*x+cx;
        float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF1->IsInImage(u,v))
            continue;

        float maxDistance = pMP->GetMaxDistanceInvariance();
        float minDistance = pMP->GetMinDistanceInvariance();
        float dist3D = cv::norm(p3Dc1);

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Compute predicted octave
        float ratio = dist3D/minDistance;

        vector<float>::iterator it = lower_bound(vfScaleFactors1.begin(), vfScaleFactors1.end(), ratio);
        const int nPredictedLevel = min(static_cast<int>(it-vfScaleFactors1.begin()),nMaxLevel1);



        // Search in a radius of 2.5*sigma(ScaleLevel)
        float radius = th*vfScaleFactors1[nPredictedLevel];
        //取出KF1中可能匹配的点
        vector<size_t> vIndices = pKF1->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        cv::Mat dMP = pMP->GetDescriptor();
        //同样寻找最佳匹配
        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            size_t idx = *vit;

            cv::KeyPoint kp = pKF1->GetKeyPointUn(idx);

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            cv::Mat dKF = pKF1->GetDescriptor(idx);

            int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch2[i2]=bestIdx;
        }
    }

    // Check agreement
    int nFound = 0;

    for(int i1=0; i1<N1; i1++)
    {
        int idx2 = vnMatch1[i1];

        if(idx2>=0)
        {
            int idx1 = vnMatch2[idx2];
            //判断是否是相互最佳匹配
            if(idx1==i1)
            {
                vpMatches12[i1] = vpMapPoints2[idx2];
                nFound++;
            }
        }
    }

    return nFound;
}
//1.将上一帧中匹配到的所有MapPoint重投影到当前帧中
//2.在以重投影坐标为中心的区域窗搜索匹配
//3.根据角度旋转一致性再剔除一些匹配点
int ORBmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, float th)
{
    //记录匹配到的数量
    int nmatches = 0;

    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;
    //取出根据运动模型预测的当前帧位姿
    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);

    for(size_t i=0, iend=LastFrame.mvpMapPoints.size(); i<iend; i++)
    {
        MapPoint* pMP = LastFrame.mvpMapPoints[i];

        if(pMP)
        {
            if(!LastFrame.mvbOutlier[i])
            {
                // Project
                //将上一帧的MapPoint投影到当前帧图像上
                cv::Mat x3Dw = pMP->GetWorldPos();
                cv::Mat x3Dc = Rcw*x3Dw+tcw;

                const float xc = x3Dc.at<float>(0);
                const float yc = x3Dc.at<float>(1);
                const float invzc = 1.0/x3Dc.at<float>(2);

                float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx;
                float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy;
                //判断是否在可视区域内
                if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)
                    continue;
                if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
                    continue;
                //此MapPoint对应的关键点所在level
                int nPredictedOctave = LastFrame.mvKeys[i].octave;

                // Search in a window. Size depends on scale
                //尺度越大，即level越高，搜索区域越大
                float radius = th*CurrentFrame.mvScaleFactors[nPredictedOctave];
                //取出当前帧中可能与前一帧MapPoint匹配的关键点索引值
                vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nPredictedOctave-1, nPredictedOctave+1);

                if(vIndices2.empty())
                    continue;

                cv::Mat dMP = LastFrame.mDescriptors.row(i);

                int bestDist = INT_MAX;
                int bestIdx2 = -1;

                for(vector<size_t>::iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
                {
                    size_t i2 = *vit;
                    //如果当前帧中可能的关键点已经有匹配的MapPoint了，pass
                    if(CurrentFrame.mvpMapPoints[i2])
                        continue;
                    //计算最小距离
                    cv::Mat d = CurrentFrame.mDescriptors.row(i2);

                    int dist = DescriptorDistance(dMP,d);

                    if(dist<bestDist)
                    {
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }

                if(bestDist<=TH_HIGH)
                {
                    //符合阈值要求就添加到当前帧中
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                    nmatches++;
                    //添加到bin中
                    if(mbCheckOrientation)
                    {
                        //上一帧与当前帧关键点角度差值
                        float rot = LastFrame.mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        ROS_ASSERT(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }
            }
        }
    }  

   //Apply rotation consistency
   //利用旋转角度一致性剔除一些点
   if(mbCheckOrientation)
   {
       int ind1=-1;
       int ind2=-1;
       int ind3=-1;

       ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

       for(int i=0; i<HISTO_LENGTH; i++)
       {
           if(i!=ind1 && i!=ind2 && i!=ind3)
           {
               for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
               {
                   CurrentFrame.mvpMapPoints[rotHist[i][j]]=NULL;
                   nmatches--;
               }
           }
       }
   }

   return nmatches;
}
//遍历该关键帧中记录的未匹配的MapPoint，重投影到当前帧再次匹配
int ORBmatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint*> &sAlreadyFound, float th ,int ORBdist)
{
    int nmatches = 0;

    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);
    //相机在世界坐标系下的坐标
    const cv::Mat Ow = -Rcw.t()*tcw;

    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();
    //遍历该关键帧中记录的MapPoint
    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP)
        {
            //判断此MapPoint是否已经和当前帧匹配了
            if(!pMP->isBad() && !sAlreadyFound.count(pMP))
            {
                //Project
                //重投影到当前帧图像上
                cv::Mat x3Dw = pMP->GetWorldPos();
                cv::Mat x3Dc = Rcw*x3Dw+tcw;

                const float xc = x3Dc.at<float>(0);
                const float yc = x3Dc.at<float>(1);
                const float invzc = 1.0/x3Dc.at<float>(2);

                float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx;
                float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy;
                //检查是否可视
                if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)
                    continue;
                if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
                    continue;

                // Compute predicted scale level
                float minDistance = pMP->GetMinDistanceInvariance();
                cv::Mat PO = x3Dw-Ow;
                float dist3D = cv::norm(PO);
                float ratio = dist3D/minDistance;

                vector<float>::iterator it = lower_bound(CurrentFrame.mvScaleFactors.begin(), CurrentFrame.mvScaleFactors.end(), ratio);
                const int nPredictedLevel = min(static_cast<int>(it-CurrentFrame.mvScaleFactors.begin()),CurrentFrame.mnScaleLevels-1);

                // Search in a window
                float radius = th*CurrentFrame.mvScaleFactors[nPredictedLevel];

                vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nPredictedLevel-1, nPredictedLevel+1);

                if(vIndices2.empty())
                    continue;

                cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = INT_MAX;
                int bestIdx2 = -1;

                for(vector<size_t>::iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
                {
                    size_t i2 = *vit;
                    if(CurrentFrame.mvpMapPoints[i2])
                        continue;

                    cv::Mat d = CurrentFrame.mDescriptors.row(i2);

                    int dist = DescriptorDistance(dMP,d);

                    if(dist<bestDist)
                    {
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }

                if(bestDist<=ORBdist)
                {
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                    nmatches++;

                    if(mbCheckOrientation)
                    {
                        float rot = pKF->GetKeyPointUn(i).angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        ROS_ASSERT(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }

            }
        }
    }


   if(mbCheckOrientation)
   {
       int ind1=-1;
       int ind2=-1;
       int ind3=-1;

       ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

       for(int i=0; i<HISTO_LENGTH; i++)
       {
           if(i!=ind1 && i!=ind2 && i!=ind3)
           {
               for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
               {
                   CurrentFrame.mvpMapPoints[rotHist[i][j]]=NULL;
                   nmatches--;
               }
           }
       }
   }

    return nmatches;
}

void ORBmatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0;
    int max2=0;
    int max3=0;

    for(int i=0; i<L; i++)
    {
        const int s = histo[i].size();
        if(s>max1)
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)
        {
            max3=s;
            ind3=i;
        }
    }

    if(max2<0.1f*(float)max1)
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1)
    {
        ind3=-1;
    }
}


// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

} //namespace ORB_SLAM
