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

#include "MapPoint.h"
#include "ORBmatcher.h"
#include "ros/ros.h"

namespace ORB_SLAM
{

long unsigned int MapPoint::nNextId=0;


MapPoint::MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map* pMap):
    mnFirstKFid(pRefKF->mnId), mnTrackReferenceForFrame(0), mnLastFrameSeen(0), mnBALocalForKF(0),
    mnLoopPointForKF(0), mnCorrectedByKF(0),mnCorrectedReference(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1),
    mbBad(false), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap)
{
    //复制3D坐标
    Pos.copyTo(mWorldPos);
    mnId=nNextId++;
    mNormalVector = cv::Mat::zeros(3,1,CV_32F);
}

void MapPoint::SetWorldPos(const cv::Mat &Pos)
{
    boost::mutex::scoped_lock lock(mMutexPos);
    Pos.copyTo(mWorldPos);
}

cv::Mat MapPoint::GetWorldPos()
{
    boost::mutex::scoped_lock lock(mMutexPos);
    return mWorldPos.clone();
}

cv::Mat MapPoint::GetNormal()
{
    boost::mutex::scoped_lock lock(mMutexPos);
    return mNormalVector.clone();
}

KeyFrame* MapPoint::GetReferenceKeyFrame()
{
     boost::mutex::scoped_lock lock(mMutexFeatures);
     return mpRefKF;
}

void MapPoint::AddObservation(KeyFrame* pKF, size_t idx)
{
    boost::mutex::scoped_lock lock(mMutexFeatures);
    mObservations[pKF]=idx;
}
//1.清除对观测到该MapPoint的关键帧的记录，如果剩余的观测<=2，清除此MapPoint，
//2.并清除所有观测到该MapPoint的关键帧中的记录，最后从Map中清除
void MapPoint::EraseObservation(KeyFrame* pKF)
{
    bool bBad=false;
    {
        boost::mutex::scoped_lock lock(mMutexFeatures);
        //删除关键帧pKF对自己的观测记录
        if(mObservations.count(pKF))
        {
            mObservations.erase(pKF);
            //如果pKF是自己的参考帧，直接取观测中的第一个作为自己的参考帧
            if(mpRefKF==pKF)
                mpRefKF=mObservations.begin()->first;

            // If only 2 observations or less, discard point
            //如果此点剩余的观测<=2，则删除此MapPoint
            if(mObservations.size()<=2)
                bBad=true;
        }
    }

    if(bBad)
        //1.清空MapPoint中的观测记录，并清除观测到该MapPoint的关键帧中的记录
        //2.从Map中清除
        SetBadFlag();
}

map<KeyFrame*, size_t> MapPoint::GetObservations()
{
    boost::mutex::scoped_lock lock(mMutexFeatures);
    return mObservations;
}

int MapPoint::Observations()
{
    boost::mutex::scoped_lock lock(mMutexFeatures);
    return mObservations.size();
}
//1.清空MapPoint中的观测记录，并清除观测到该MapPoint的关键帧中的记录
//2.从Map中清除
void MapPoint::SetBadFlag()
{
    map<KeyFrame*,size_t> obs;
    {
        boost::mutex::scoped_lock lock1(mMutexFeatures);
        boost::mutex::scoped_lock lock2(mMutexPos);
        mbBad=true;
        obs = mObservations;
        //清空该MapPoint的观测关键帧
        mObservations.clear();
    }
    //将观测到该MapPoint的所有关键帧的记录都清楚，置为NULL
    for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        pKF->EraseMapPointMatch(mit->second);
    }
    //将此MapPoint从Map中删除
    mpMap->EraseMapPoint(this);
}

void MapPoint::Replace(MapPoint* pMP)
{
    if(pMP->mnId==this->mnId)
        return;

    map<KeyFrame*,size_t> obs;
    {
        boost::mutex::scoped_lock lock1(mMutexFeatures);
        boost::mutex::scoped_lock lock2(mMutexPos);
        obs=mObservations;
        mObservations.clear();
        //原始MapPoint设置为mbBad=true
        mbBad=true;
    }

    for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        // Replace measurement in keyframe
        KeyFrame* pKF = mit->first;
        //判断新的MapPoint中是否已经被该关键帧观测到了，如果没有，则用新的MapPoint替换原来的
        //换言之，将其他关键帧的观测都换成新的MapPoint
        if(!pMP->IsInKeyFrame(pKF))
        {
            pKF->ReplaceMapPointMatch(mit->second, pMP);
            pMP->AddObservation(pKF,mit->second);
        }
        //如果该关键帧已经观测到了新产生的MapPoint，清除对此MapPoint的观测记录
        else
        {
            pKF->EraseMapPointMatch(mit->second);
        }
    }

    pMP->ComputeDistinctiveDescriptors();
    //从地图中删除旧的MapPoint
    mpMap->EraseMapPoint(this);

}

bool MapPoint::isBad()
{
    boost::mutex::scoped_lock lock(mMutexFeatures);
    boost::mutex::scoped_lock lock2(mMutexPos);
    return mbBad;
}

void MapPoint::IncreaseVisible()
{
    boost::mutex::scoped_lock lock(mMutexFeatures);
    mnVisible++;
}

void MapPoint::IncreaseFound()
{
    boost::mutex::scoped_lock lock(mMutexFeatures);
    mnFound++;
}

float MapPoint::GetFoundRatio()
{
    boost::mutex::scoped_lock lock(mMutexFeatures);
    return static_cast<float>(mnFound)/mnVisible;
}
//同一个MapPoint可以被很多关键帧观测到，而描述此MapPoint的描述子只用一个
//因此从众多的对应的关键点的描述子中选择一个距离其他关键点描述子都相对比较近的
//作为该MapPoint的描述子
void MapPoint::ComputeDistinctiveDescriptors()
{
    // Retrieve all observed descriptors
    vector<cv::Mat> vDescriptors;

    map<KeyFrame*,size_t> observations;

    {
        boost::mutex::scoped_lock lock1(mMutexFeatures);
        if(mbBad)
            return;
        //mObservations存储以关键帧指针为key，以此关键帧的关键点索引为value
        observations=mObservations;
    }

    if(observations.empty())
        return;

    vDescriptors.reserve(observations.size());

    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;

        if(!pKF->isBad())
            //将观测到此MapPoint对应的所有描述子都取出来，放入vDescriptors
            vDescriptors.push_back(pKF->GetDescriptor(mit->second));
    }

    if(vDescriptors.empty())
        return;

    // Compute distances between them
    const size_t N = vDescriptors.size();

    float Distances[N][N];
    for(size_t i=0;i<N;i++)
    {
        //对角线上的全置0，因为自己与自己的距离肯定是0
        Distances[i][i]=0;
        for(size_t j=i+1;j<N;j++)
        {
            //计算非对角线上的距离，f1和f2的距离与f2和f1的距离相等，所以求一次之后对称写入距离
            int distij = ORBmatcher::DescriptorDistance(vDescriptors[i],vDescriptors[j]);
            Distances[i][j]=distij;
            Distances[j][i]=distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    int BestMedian = INT_MAX;
    int BestIdx = 0;
    for(size_t i=0;i<N;i++)
    {
        //取出距离方阵中的一行
        vector<int> vDists(Distances[i],Distances[i]+N);
        //从小到大排序
        sort(vDists.begin(),vDists.end());
        //取出这一行距离排在中间位置的那个：获得中值
        int median = vDists[0.5*(N-1)];
        //获得最小的中值
        if(median<BestMedian)
        {
            BestMedian = median;
            BestIdx = i;
        }
    }
    
    // 最好的描述子，该描述子相对于其他描述子有最小的距离中值
    // 简化来讲，中值代表了这个描述子到其它描述子的平均距离
    // 最好的描述子就是和其它描述子的平均距离最小
    {
        boost::mutex::scoped_lock lock(mMutexFeatures);
        mDescriptor = vDescriptors[BestIdx].clone();       
    }
}

cv::Mat MapPoint::GetDescriptor()
{
    boost::mutex::scoped_lock lock(mMutexFeatures);
    return mDescriptor.clone();
}

int MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
{
    boost::mutex::scoped_lock lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return mObservations[pKF];
    else
        return -1;
}

bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
{
    boost::mutex::scoped_lock lock(mMutexFeatures);
    return (mObservations.count(pKF));
}

void MapPoint::UpdateNormalAndDepth()
{
    map<KeyFrame*,size_t> observations;
    KeyFrame* pRefKF;
    cv::Mat Pos;
    {
        boost::mutex::scoped_lock lock1(mMutexFeatures);
        boost::mutex::scoped_lock lock2(mMutexPos);
        if(mbBad)
            return;
        observations=mObservations;
        pRefKF=mpRefKF;
        Pos = mWorldPos.clone();
    }

    cv::Mat normal = cv::Mat::zeros(3,1,CV_32F);
    int n=0;
    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        //当前相机在世界坐标系下的3D位置
        cv::Mat Owi = pKF->GetCameraCenter();
        cv::Mat normali = mWorldPos - Owi;
        normal = normal + normali/cv::norm(normali);
        n++;
    } 

    cv::Mat PC = Pos - pRefKF->GetCameraCenter();
    //MapPoint与参考关键帧相机的距离
    const float dist = cv::norm(PC);
    //参考帧对应的描述子所在level
    const int level = pRefKF->GetKeyPointScaleLevel(observations[pRefKF]);
    //获得图像尺度因子，因为level=0尺度为1.0，所以level=1的尺度正好等于尺度因子，
    //作为尺度因子所以才获取level=1的尺度
    const float scaleFactor = pRefKF->GetScaleFactor();
    //对应描述子对应的所在尺度
    const float levelScaleFactor =  pRefKF->GetScaleFactor(level);
    const int nLevels = pRefKF->GetScaleLevels();

    {
        boost::mutex::scoped_lock lock3(mMutexPos);
        mfMinDistance = (1.0f/scaleFactor)*dist / levelScaleFactor;
        mfMaxDistance = scaleFactor*dist * pRefKF->GetScaleFactor(nLevels-1-level);
        //所有关键帧观测到此MapPoint的平均向量
        mNormalVector = normal/n;
    }
}

float MapPoint::GetMinDistanceInvariance()
{
    boost::mutex::scoped_lock lock(mMutexPos);
    return mfMinDistance;
}

float MapPoint::GetMaxDistanceInvariance()
{
    boost::mutex::scoped_lock lock(mMutexPos);
    return mfMaxDistance;
}

} //namespace ORB_SLAM
