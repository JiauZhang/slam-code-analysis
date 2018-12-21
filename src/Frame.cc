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

#include "Frame.h"
#include "Converter.h"

#include <ros/ros.h>

namespace ORB_SLAM
{
long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy;
int Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame()
{}

//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractor(frame.mpORBextractor), im(frame.im.clone()), mTimeStamp(frame.mTimeStamp),
     mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()), N(frame.N), mvKeys(frame.mvKeys), mvKeysUn(frame.mvKeysUn),
     mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec), mDescriptors(frame.mDescriptors.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier),
     mnId(frame.mnId),
     mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels), mfScaleFactor(frame.mfScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];

    if(!frame.mTcw.empty())
        mTcw = frame.mTcw.clone();
}

//输入灰度图，时间戳，特征提取器
//把灰度图构建成图像金字塔，并对每个level进行特征点提取和描述子提取
//并将关键点分配到网格中，且每个网格记录落入的特征点的索引值
Frame::Frame(cv::Mat &im_, const double &timeStamp, ORBextractor* extractor, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef)
    :mpORBvocabulary(voc),mpORBextractor(extractor), im(im_),mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone())
{
    // Exctract ORB  
    //提取每个level的关键点，并将关键点坐标都放大到level=0上对应的坐标
    //描述子根据关键点所在尺度的图像金字塔对应尺度进行提取
    (*mpORBextractor)(im,cv::Mat(),mvKeys,mDescriptors);
    //提取到的关键点总数
    N = mvKeys.size();

    if(mvKeys.empty())
        return;
    //初始化Frame的mvpMapPoints，使其与mvKeys大小相同，都初始化为NULL
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    //根据校正参数校正关键点
    UndistortKeyPoints();


    // This is done for the first created Frame
    //只有系统处理的第一帧图像执行此条件分支
    if(mbInitialComputations)
    {
        ComputeImageBounds();

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);

        mbInitialComputations=false;
    }


    mnId=nNextId++;    

    //Scale Levels Info
    mnScaleLevels = mpORBextractor->GetLevels();
    mfScaleFactor = mpORBextractor->GetScaleFactor();

    mvScaleFactors.resize(mnScaleLevels);
    mvLevelSigma2.resize(mnScaleLevels);
    //level=0的缩放因子为1.0
    mvScaleFactors[0]=1.0f;
    //level=0的信息矩阵方差为1.0
    mvLevelSigma2[0]=1.0f;
    //填充每个level的尺度因子和信息矩阵方差
    for(int i=1; i<mnScaleLevels; i++)
    {
        mvScaleFactors[i]=mvScaleFactors[i-1]*mfScaleFactor;        
        mvLevelSigma2[i]=mvScaleFactors[i]*mvScaleFactors[i];
    }

    mvInvLevelSigma2.resize(mvLevelSigma2.size());
    //信息矩阵方差的逆
    for(int i=0; i<mnScaleLevels; i++)
        mvInvLevelSigma2[i]=1/mvLevelSigma2[i];

    // Assign Features to Grid Cells
    // N 是提取到的关键点总数
    //mGrid用来存储落入每个cell中的关键点的index
    int nReserve = 0.5*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);

    //记录落入每个网格中的关键点的索引值
    for(size_t i=0;i<mvKeysUn.size();i++)
    {
        cv::KeyPoint &kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }

    //用作记录为outlier的flag，默认全为false
    mvbOutlier = vector<bool>(N,false);

}

void Frame::UpdatePoseMatrices()
{ 
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mtcw = mTcw.rowRange(0,3).col(3);
    //为什么要加个负号呢？？？ 2018-12-13
    mOw = -mRcw.t()*mtcw;
}
//判断mvpLocalMapPoints中未与当前帧匹配的MapPoint是否可能在当前帧中观测到
//并设置相应的变量mbTrackInView=true指示出来
bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos(); 

    // 3D in camera coordinates
    //该MapPoint在当前帧相机坐标系下的坐标
    const cv::Mat Pc = mRcw*P+mtcw;
    const float PcX = Pc.at<float>(0);
    const float PcY= Pc.at<float>(1);
    const float PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0)
        return false;

    // Project in image and check it is not outside
    //将Pc相机坐标投影到图像坐标(u,v)
    const float invz = 1.0/PcZ;
    const float u=fx*PcX*invz+cx;
    const float v=fy*PcY*invz+cy;
    //判断是否在畸变矫正后的区域范围内，因为所有的MapPoint都是矫正后的关键点得到的
    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    //MapPoint与相机中心的距离
    const cv::Mat PO = P-mOw;
    const float dist = cv::norm(PO);

    if(dist<minDistance || dist>maxDistance)
        return false;

   // Check viewing angle
    cv::Mat Pn = pMP->GetNormal();
    //余弦公式
    float viewCos = PO.dot(Pn)/dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale level acording to the distance
    float ratio = dist/minDistance;

    vector<float>::iterator it = lower_bound(mvScaleFactors.begin(), mvScaleFactors.end(), ratio);
    int nPredictedLevel = it-mvScaleFactors.begin();

    if(nPredictedLevel>=mnScaleLevels)
        nPredictedLevel=mnScaleLevels-1;

    // Data used by the tracking
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = u;
    pMP->mTrackProjY = v;
    pMP->mnTrackScaleLevel= nPredictedLevel;
    pMP->mTrackViewCos = viewCos;

    return true;
}

/**
 * @brief 找到在 以x,y为中心,边长为2r的方形内且在[minLevel, maxLevel]的特征点
 * @param x        图像坐标u
 * @param y        图像坐标v
 * @param r        边长
 * @param minLevel 最小尺度
 * @param maxLevel 最大尺度
 * @return         满足条件的特征点的序号
 */
 //1.根据给定坐标和区域半径计算出所在网格范围
 //2.在网格中取出符合level要求的关键点
vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, int minLevel, int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(mvKeysUn.size());
    //floor向下取整，round四舍五入
    //mfGridElementWidthInv=(FRAME_GRID_COLS)/(mnMaxX-mnMinX)
    //计算给定关键点往左 r 所属的网格的左边界
    int nMinCellX = floor((x-mnMinX-r)*mfGridElementWidthInv);
    nMinCellX = max(0,nMinCellX);
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;
    //计算给定关键点往右 r 所属的网格的右边界
    int nMaxCellX = ceil((x-mnMinX+r)*mfGridElementWidthInv);
    nMaxCellX = min(FRAME_GRID_COLS-1,nMaxCellX);
    if(nMaxCellX<0)
        return vIndices;
    //上边界
    int nMinCellY = floor((y-mnMinY-r)*mfGridElementHeightInv);
    nMinCellY = max(0,nMinCellY);
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;
    //下边界
    int nMaxCellY = ceil((y-mnMinY+r)*mfGridElementHeightInv);
    nMaxCellY = min(FRAME_GRID_ROWS-1,nMaxCellY);
    if(nMaxCellY<0)
        return vIndices;

    bool bCheckLevels=true;
    bool bSameLevel=false;
    if(minLevel==-1 && maxLevel==-1)
        bCheckLevels=false;
    else
        if(minLevel==maxLevel)
            bSameLevel=true;
    //遍历所有被选择的网格
    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            vector<size_t> vCell = mGrid[ix][iy];
            //如果网格中不包含关键点，则开始下一个网格
            if(vCell.empty())
                continue;
            //遍历一个网格中所有的关键点索引的关键点
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                //取出网格记录的索引对应的关键点
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                if(bCheckLevels && !bSameLevel)
                {
                    //当对level有要求时，检查是否符合level的要求，不符合就进行下一个
                    if(kpUn.octave<minLevel || kpUn.octave>maxLevel)
                        continue;
                }
                else if(bSameLevel)
                {
                    //当仅在某一层查找时，检测是否是对应level的关键点，不符合进行下一个
                    if(kpUn.octave!=minLevel)
                        continue;
                }
                //检查关键点是否在指定搜索半径内，因为网格的选取是按包含搜索区域来选取的
                //因此会包含搜索区域半径r之外的关键点
                if(abs(kpUn.pt.x-x)>r || abs(kpUn.pt.y-y)>r)
                    continue;
                //符合所有要求之后，记录关键点索引值
                vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;

}

bool Frame::PosInGrid(cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}


void Frame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        //将描述子转化为向量的组合
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}
//根据校正参数校正关键点
void Frame::UndistortKeyPoints()
{
    //如果没有畸变，则直接让mvKeysUn=mvKeys
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(mvKeys.size(),2,CV_32F);
    //将mvKeys包含的坐标值(u,v)转换为Mat结构
    for(unsigned int i=0; i<mvKeys.size(); i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    //获得畸变校正后的点
    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    mvKeysUn.resize(mvKeys.size());
    //将Mat结构中存放的矫正后的点放入mvKeysUn，可以看出，mvKeysUn和mvKeys的size是相同的
    for(unsigned int i=0; i<mvKeys.size(); i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}

void Frame::ComputeImageBounds()
{
    //当相机有畸变时执行次分支
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;//左上
        mat.at<float>(1,0)=im.cols; mat.at<float>(1,1)=0.0;//右上
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=im.rows;//左下
        mat.at<float>(3,0)=im.cols; mat.at<float>(3,1)=im.rows;//右下

        // Undistort corners
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);
        //MinX根据左下和左上的x求出
        mnMinX = min(floor(mat.at<float>(0,0)),floor(mat.at<float>(2,0)));
        //MaxX根据右下和右上的x求出
        mnMaxX = max(ceil(mat.at<float>(1,0)),ceil(mat.at<float>(3,0)));
        //方法同上
        mnMinY = min(floor(mat.at<float>(0,1)),floor(mat.at<float>(1,1)));
        mnMaxY = max(ceil(mat.at<float>(2,1)),ceil(mat.at<float>(3,1)));

    }
    //无畸变时边界直接设置为图像大小
    else
    {
        mnMinX = 0;
        mnMaxX = im.cols;
        mnMinY = 0;
        mnMaxY = im.rows;
    }
}

} //namespace ORB_SLAM
