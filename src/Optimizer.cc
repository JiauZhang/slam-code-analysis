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

#include "Optimizer.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include<Eigen/StdVector>

#include "Converter.h"

namespace ORB_SLAM
{

void Optimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations, bool* pbStopFlag)
{
    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    vector<MapPoint*> vpMP = pMap->GetAllMapPoints();
    BundleAdjustment(vpKFs,vpMP,nIterations,pbStopFlag);
}

//同时优化关键帧位姿和MapPoint的坐标
void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP, int nIterations, bool* pbStopFlag)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;

    // SET KEYFRAME VERTICES
    //关键帧位姿
    for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
        vSE3->setId(pKF->mnId);
        vSE3->setFixed(pKF->mnId==0);
        optimizer.addVertex(vSE3);
        if(pKF->mnId>maxKFid)
            maxKFid=pKF->mnId;
    }


    const float thHuber = sqrt(5.991);

    // SET MAP POINT VERTICES
    //MapPoint的坐标也被优化
    for(size_t i=0, iend=vpMP.size(); i<iend;i++)
    {
        MapPoint* pMP = vpMP[i];
        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        map<KeyFrame*,size_t> observations = pMP->GetObservations();

        //SET EDGES
        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKF = mit->first;
            if(pKF->isBad())
                continue;
            Eigen::Matrix<double,2,1> obs;
            cv::KeyPoint kpUn = pKF->GetKeyPointUn(mit->second);
            obs << kpUn.pt.x, kpUn.pt.y;

            g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
            e->setMeasurement(obs);
            float invSigma2 = pKF->GetInvSigma2(kpUn.octave);
            e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(thHuber);

            e->fx = pKF->fx;
            e->fy = pKF->fy;
            e->cx = pKF->cx;
            e->cy = pKF->cy;

            optimizer.addEdge(e);
        }
    }

    // Optimize!

    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);

    // Recover optimized data

    //Keyframes
    //将优化后的关键帧位姿更新到关键帧数据结构中
    for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
    {
        KeyFrame* pKF = vpKFs[i];
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKF->SetPose(Converter::toCvMat(SE3quat));
    }

    //Points
    //更新MapPoint的坐标
    for(size_t i=0, iend=vpMP.size(); i<iend;i++)
    {
        MapPoint* pMP = vpMP[i];
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }

}
//根据当前帧中匹配到的MapPoint优化相机位姿
//误差定义为MapPoint投影到当前帧的像素坐标与关键点坐标的差值
//返回除了outlier外的MapPoint数量
int Optimizer::PoseOptimization(Frame *pFrame)
{    
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    optimizer.setVerbose(false);
    //记录总共添加了多少个MapPoint
    int nInitialCorrespondences=0;

    // SET FRAME VERTEX
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    //相机位姿初值是直接复制前一帧的位姿得到的
    //添加顶点，此处就一个顶点，即当前帧的相机位姿
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // SET MAP POINT VERTICES
    //虽然此处把MapPoint也添加为顶点了，但是后边设置了MapPoint不进行优化
    vector<g2o::EdgeSE3ProjectXYZ*> vpEdges;
    vector<g2o::VertexSBAPointXYZ*> vVertices;
    vector<float> vInvSigmas2;
    vector<size_t> vnIndexEdge;

    const int N = pFrame->mvpMapPoints.size();
    vpEdges.reserve(N);
    vVertices.reserve(N);
    vInvSigmas2.reserve(N);
    vnIndexEdge.reserve(N);

    const float delta = sqrt(5.991);
    //
    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = pFrame->mvpMapPoints[i];
        if(pMP)
        {
            g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
            //添加MapPoint顶点
            vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
            vPoint->setId(i+1);
            //不优化此顶点
            vPoint->setFixed(true);
            optimizer.addVertex(vPoint);
            vVertices.push_back(vPoint);
            //记录总共添加了多少个MapPoint
            nInitialCorrespondences++;
            pFrame->mvbOutlier[i] = false;

            //SET EDGE
            Eigen::Matrix<double,2,1> obs;
            //MapPoint对应的观测值，即mvKeysUn存放的关键点坐标值
            cv::KeyPoint kpUn = pFrame->mvKeysUn[i];
            //MapPoint对应的观测值，即像素坐标
            obs << kpUn.pt.x, kpUn.pt.y;

            g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(i+1)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
            e->setMeasurement(obs);
            const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
            e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(delta);

            e->fx = pFrame->fx;
            e->fy = pFrame->fy;
            e->cx = pFrame->cx;
            e->cy = pFrame->cy;

            e->setLevel(0);

            optimizer.addEdge(e);

            vpEdges.push_back(e);
            vInvSigmas2.push_back(invSigma2);
            vnIndexEdge.push_back(i);
        }

    }

    // We perform 4 optimizations, decreasing the inlier region
    // From second to final optimization we include only inliers in the optimization
    // At the end of each optimization we check which points are inliers
    //判断是否为inliers的误差阈值
    const float chi2[4]={9.210,7.378,5.991,5.991};
    //这4次优化的迭代次数
    const int its[4]={10,10,7,5};

    int nBad=0;
    for(size_t it=0; it<4; it++)
    {
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);
        //记录最后一轮优化时outlier的数量
        nBad=0;
        //第一次优化之后开始评判现有的顶点是否为outliers，并记录数量
        for(size_t i=0, iend=vpEdges.size(); i<iend; i++)
        {
            g2o::EdgeSE3ProjectXYZ* e = vpEdges[i];

            const size_t idx = vnIndexEdge[i];
            //mvbOutlier初始值都为false
            if(pFrame->mvbOutlier[idx])
                e->computeError();

            if(e->chi2()>chi2[it])
            {
                //如果误差过大，就视为outlier
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else if(e->chi2()<=chi2[it])
            {
                //如果误差小于阈值，则认为是inlier
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
            }
        }

        if(optimizer.edges().size()<10)
            break;
    }

    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3quat_recov);
    //将优化后的相机位姿跟新到当前帧中
    pose.copyTo(pFrame->mTcw);
    //返回除了outlier外的MapPoint数量
    return nInitialCorrespondences-nBad;
}
//1.取出当前关键帧的所有共视帧
//2.再取出和共视关键帧有共识MapPoint的关键帧
//3.进行g2o优化，共视关键帧和所有MapPoint被优化，第0帧和其他关键帧仅提供约束，不进行位姿优化
void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag)
{    
    // Local KeyFrames: First Breath Search from Current Keyframe
    list<KeyFrame*> lLocalKeyFrames;

    lLocalKeyFrames.push_back(pKF);
    //记录localBA的当前关键帧Id
    pKF->mnBALocalForKF = pKF->mnId;
    //取当前关键帧的共视帧
    vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    //把当前帧共视帧中可用的都记录到lLocalKeyFrames中
    for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
    {
        KeyFrame* pKFi = vNeighKFs[i];
        //这些关键帧都记录下进行LocalBA时的当前关键帧Id
        pKFi->mnBALocalForKF = pKF->mnId;
        if(!pKFi->isBad())
            lLocalKeyFrames.push_back(pKFi);
    }

    // Local MapPoints seen in Local KeyFrames
    //把lLocalKeyFrames中MapPoint都取出来记录到lLocalMapPoints中
    list<MapPoint*> lLocalMapPoints;
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
                if(!pMP->isBad())
                    //只取没有被当前帧取出过的MapPoint
                    if(pMP->mnBALocalForKF!=pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF=pKF->mnId;
                    }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    //取出那些没有被当前帧加入共视关系，但存在与当前帧的共视帧看到的相同的MapPoint的关键帧
    //简言之，与当前帧没有共视，但与当前帧的共视帧存在共视的关键帧
    list<KeyFrame*> lFixedCameras;
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        map<KeyFrame*,size_t> observations = (*lit)->GetObservations();
        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
            {
                //该关键帧记录当前帧Id
                pKFi->mnBAFixedForKF=pKF->mnId;
                if(!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;

    // SET LOCAL KEYFRAME VERTICES
    //将lLocalKeyFrames中所有的关键帧加入到Vertex中
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        //除了第0帧关键帧固定之外，因为第0关键帧是世界坐标系，其他的都设置为可优化量
        vSE3->setFixed(pKFi->mnId==0);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }

    // SET FIXED KEYFRAME VERTICES
    //共视的共视帧也都添加到Vertex中，但是都不加入优化，仅提供约束
    for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        //不进行优化
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }

    // SET MAP POINT VERTICES
    //此值是一个最大估值，因为不可能所有的MapPoint都和每个关键帧都完全相连
    const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size();

    vector<g2o::EdgeSE3ProjectXYZ*> vpEdges;
    vpEdges.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKF;
    vpEdgeKF.reserve(nExpectedSize);

    vector<float> vSigmas2;
    vSigmas2.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdge;
    vpMapPointEdge.reserve(nExpectedSize);

    const float thHuber = sqrt(5.991);
    //把所有的lLocalMapPoints添加到Vertex，同时把与其相连的关键帧添加为Edge
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        //这个什么时候设置值得研究一下
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        map<KeyFrame*,size_t> observations = pMP->GetObservations();

        //SET EDGES
        //将所有观测到该MapPoint的关键帧与该MapPoint组成边
        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(!pKFi->isBad()) //这句是废话，添加关键帧的时候已经验证过了，也许是因为多线程的问题吧，anyway~
            {
                //边是重投影误差的函数
                Eigen::Matrix<double,2,1> obs;
                cv::KeyPoint kpUn = pKFi->GetKeyPointUn(mit->second);
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
                //0->MapPoint
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                //1->KeyFrame
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                e->setMeasurement(obs);
                float sigma2 = pKFi->GetSigma2(kpUn.octave);
                float invSigma2 = pKFi->GetInvSigma2(kpUn.octave);
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuber);

                e->fx = pKFi->fx;
                e->fy = pKFi->fy;
                e->cx = pKFi->cx;
                e->cy = pKFi->cy;

                optimizer.addEdge(e);
                vpEdges.push_back(e);
                vpEdgeKF.push_back(pKFi);
                vSigmas2.push_back(sigma2);
                vpMapPointEdge.push_back(pMP);
            }
        }
    }

    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // Check inlier observations
    //经过第一轮优化后，剔除一些不好的MapPoint
    for(size_t i=0, iend=vpEdges.size(); i<iend;i++)
    {
        g2o::EdgeSE3ProjectXYZ* e = vpEdges[i];
        MapPoint* pMP = vpMapPointEdge[i];

        if(pMP->isBad())
            continue;
        //阈值如何设定的？？ -2018-12-15
        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKF[i];
            pKFi->EraseMapPointMatch(pMP);
            pMP->EraseObservation(pKFi);

            optimizer.removeEdge(e);
            vpEdges[i]=NULL;
        }
    }

    // Recover optimized data

    //Keyframes
    //将第一轮优化后的位姿更新到关键帧中
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKF = *lit;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKF->SetPose(Converter::toCvMat(SE3quat));
    }
    //Points
    //将第一轮优化后的MapPoint的3D坐标更新，并更新UpdateNormalAndDepth
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }

    // Optimize again without the outliers

    optimizer.initializeOptimization();
    optimizer.optimize(10);

    // Check inlier observations
    for(size_t i=0, iend=vpEdges.size(); i<iend;i++)
    {
        g2o::EdgeSE3ProjectXYZ* e = vpEdges[i];

        if(!e)
            continue;

        MapPoint* pMP = vpMapPointEdge[i];

        if(pMP->isBad())
            continue;
        //阈值如何设定的？？ -2018-12-15
        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKF = vpEdgeKF[i];
            pKF->EraseMapPointMatch(pMP->GetIndexInKeyFrame(pKF));
            pMP->EraseObservation(pKF);
        }
    }

    // Recover optimized data

    //Keyframes
    //将优化后的位姿更新到关键帧中
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKF = *lit;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKF->SetPose(Converter::toCvMat(SE3quat));
    }

    //Points
    //将第一轮优化后的MapPoint的3D坐标更新，并更新UpdateNormalAndDepth
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }
}



void Optimizer::OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF, g2o::Sim3 &Scurw,
                                       LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       map<KeyFrame *, set<KeyFrame *> > &LoopConnections)
{
    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
           new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3 * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

    unsigned int nMaxKFid = pMap->GetMaxKFid();


    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);
    vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid+1);

    const int minFeat = 100;

    // SET KEYFRAME VERTICES
    //把所有关键帧都设为顶点
    for(size_t i=0, iend=vpKFs.size(); i<iend;i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        int nIDi = pKF->mnId;      
        //sim3更新过的关键帧
        if(CorrectedSim3.count(pKF))
        {
            vScw[nIDi] = CorrectedSim3[pKF];
            VSim3->setEstimate(CorrectedSim3[pKF]);
        }
        //没有被sim3更新过的转化为尺度为 1 的sim3
        else
        {
            Eigen::Matrix<double,3,3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
            Eigen::Matrix<double,3,1> tcw = Converter::toVector3d(pKF->GetTranslation());
            g2o::Sim3 Siw(Rcw,tcw,1.0);
            vScw[nIDi] = Siw;
            VSim3->setEstimate(Siw);
        }
        //和当前帧匹配上的闭环帧不进行位姿优化
        if(pKF==pLoopKF)
            VSim3->setFixed(true);

        VSim3->setId(nIDi);
        //这里没有MapPoint，所以不需要此功能
        VSim3->setMarginalized(false);

        optimizer.addVertex(VSim3);

        vpVertices[nIDi]=VSim3;
    }


    set<pair<long unsigned int,long unsigned int> > sInsertedEdges;

    Eigen::Matrix<double,7,7> matLambda = Eigen::Matrix<double,7,7>::Identity();

    // SET LOOP EDGES
    //LoopConnectioss是由于闭环新产生的共视关系连接
    for(map<KeyFrame *, set<KeyFrame *> >::iterator mit = LoopConnections.begin(), mend=LoopConnections.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        const long unsigned int nIDi = pKF->mnId;
        set<KeyFrame*> &spConnections = mit->second;
        g2o::Sim3 Siw = vScw[nIDi];
        g2o::Sim3 Swi = Siw.inverse();
        //pKF新的共视帧
        for(set<KeyFrame*>::iterator sit=spConnections.begin(), send=spConnections.end(); sit!=send; sit++)
        {
            //关键帧Id
            const long unsigned int nIDj = (*sit)->mnId;
            if((nIDi!=pCurKF->mnId || nIDj!=pLoopKF->mnId) && pKF->GetWeight(*sit)<minFeat)
                continue;
            //vScw中存放的是闭环关键帧的Sim3
            g2o::Sim3 Sjw = vScw[nIDj];
            //pKF变换到(*sit)的旋转矩阵(sim3)
            g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;

            optimizer.addEdge(e);

            sInsertedEdges.insert(make_pair(min(nIDi,nIDj),max(nIDi,nIDj)));
        }
    }

    // SET NORMAL EDGES
    for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
    {
        KeyFrame* pKF = vpKFs[i];

        const int nIDi = pKF->mnId;

        g2o::Sim3 Swi;
        if(NonCorrectedSim3.count(pKF))
            Swi = NonCorrectedSim3[pKF].inverse();
        else
            Swi = vScw[nIDi].inverse();

        KeyFrame* pParentKF = pKF->GetParent();

        // Spanning tree edge
        //先建立pKF与其父节点的边
        if(pParentKF)
        {
            int nIDj = pParentKF->mnId;

            g2o::Sim3 Sjw;
            //如果记录有没有调整过的sim3，则使用没有调整过的
            if(NonCorrectedSim3.count(pParentKF))
                Sjw = NonCorrectedSim3[pParentKF];
            else
                Sjw = vScw[nIDj];
            //子变换到父所需要的sim3
            g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // Loop edges
        //把以前与pKF形成闭环的关键帧也添加为边
        set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();
        for(set<KeyFrame*>::iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
        {
            KeyFrame* pLKF = *sit;
            //判断pKF是不是发现闭环的帧，而pLKF是被闭环的
            //即判断pKF是否是当前帧，而pLKF是闭环帧
            //如果成了，则必有pLKF->mnId<pKF->mnId
            if(pLKF->mnId<pKF->mnId)
            {
                g2o::Sim3 Slw;
                if(NonCorrectedSim3.count(pLKF))
                    Slw = NonCorrectedSim3[pLKF];
                else
                    Slw = vScw[pLKF->mnId];

                g2o::Sim3 Sli = Slw * Swi;
                g2o::EdgeSim3* el = new g2o::EdgeSim3();
                el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
                el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                el->setMeasurement(Sli);
                el->information() = matLambda;
                optimizer.addEdge(el);
            }
        }

        // Covisibility graph edges
        //把共视关系也添加为边
        //这里并不是把所有共视帧都添加进来，而是今天加符合权重要求的，即连接关系比较强的才可以
        vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
        for(vector<KeyFrame*>::iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
        {
            KeyFrame* pKFn = *vit;
            if(pKFn && pKFn!=pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))
            {
                if(!pKFn->isBad() && pKFn->mnId<pKF->mnId)
                {
                    if(sInsertedEdges.count(make_pair(min(pKF->mnId,pKFn->mnId),max(pKF->mnId,pKFn->mnId))))
                        continue;

                    g2o::Sim3 Snw;
                    if(NonCorrectedSim3.count(pKFn))
                        Snw = NonCorrectedSim3[pKFn];
                    else
                        Snw = vScw[pKFn->mnId];

                    g2o::Sim3 Sni = Snw * Swi;

                    g2o::EdgeSim3* en = new g2o::EdgeSim3();
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                }
            }
        }
    }

    // OPTIMIZE

    optimizer.initializeOptimization();
    optimizer.optimize(20);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        const int nIDi = pKFi->mnId;

        g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
        g2o::Sim3 CorrectedSiw =  VSim3->estimate();
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();
        Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
        Eigen::Vector3d eigt = CorrectedSiw.translation();
        double s = CorrectedSiw.scale();

        eigt *=(1./s); //[R t/s;0 1]

        cv::Mat Tiw = Converter::toCvSE3(eigR,eigt);

        pKFi->SetPose(Tiw);
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP->isBad())
            continue;

        int nIDr;
        if(pMP->mnCorrectedByKF==pCurKF->mnId)
        {
            nIDr = pMP->mnCorrectedReference;
        }
        else
        {
            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
            nIDr = pRefKF->mnId;
        }


        g2o::Sim3 Srw = vScw[nIDr];
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

        cv::Mat P3Dw = pMP->GetWorldPos();
        Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
        //使用未校正的位姿将MapPoint投影到相机坐标系，再用优化后的sim3投影回去
        Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

        cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
        pMP->SetWorldPos(cvCorrectedP3Dw);

        pMP->UpdateNormalAndDepth();
    }
}
//1.建立sim3顶点
//2.添加匹配的MapPoint顶点，并通过相互投影建立误差方程
//3.优化，剔除outlier，再优化
int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, float th2)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Calibration
    cv::Mat K1 = pKF1->GetCalibrationMatrix();
    cv::Mat K2 = pKF2->GetCalibrationMatrix();

    // Camera poses
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    // SET SIMILARITY VERTEX
    g2o::VertexSim3Expmap * vSim3 = new g2o::VertexSim3Expmap();
    vSim3->setEstimate(g2oS12);
    vSim3->setId(0);
    vSim3->setFixed(false);
    vSim3->_principle_point1[0] = K1.at<float>(0,2);
    vSim3->_principle_point1[1] = K1.at<float>(1,2);
    vSim3->_focal_length1[0] = K1.at<float>(0,0);
    vSim3->_focal_length1[1] = K1.at<float>(1,1);
    vSim3->_principle_point2[0] = K2.at<float>(0,2);
    vSim3->_principle_point2[1] = K2.at<float>(1,2);
    vSim3->_focal_length2[0] = K2.at<float>(0,0);
    vSim3->_focal_length2[1] = K2.at<float>(1,1);
    optimizer.addVertex(vSim3);

    // SET MAP POINT VERTICES
    //当前帧和闭环帧的匹配数量
    const int N = vpMatches1.size();
    vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    vector<g2o::EdgeSim3ProjectXYZ*> vpEdges12;
    vector<g2o::EdgeInverseSim3ProjectXYZ*> vpEdges21;
    vector<float> vSigmas12, vSigmas21;
    vector<size_t> vnIndexEdge;

    vnIndexEdge.reserve(2*N);
    vpEdges12.reserve(2*N);
    vpEdges21.reserve(2*N);

    const float deltaHuber = sqrt(th2);

    int nCorrespondences = 0;
    //添加MapPoint为顶点，不优化MapPoint
    //优化方法是
    //1.将当前帧的MapPoint投影到相机坐标系下，通过sim3.inverse投影到闭环帧图像上
    //与闭环帧对应的`MapPoint对应的关键点`像素坐标形成差值
    //2.将闭环帧的MapPoint投影到相机坐标系下，通过sim3投影到当前帧图像上
    //与当前帧对应的`MapPoint对应的关键点`像素坐标形成差值
    for(int i=0; i<N; i++)
    {
        if(!vpMatches1[i])
            continue;
        //当前帧的MapPoint
        MapPoint* pMP1 = vpMapPoints1[i];
        //闭环帧的MapPoint
        MapPoint* pMP2 = vpMatches1[i];

        int id1 = 2*i+1;
        int id2 = 2*(i+1);

        int i2 = pMP2->GetIndexInKeyFrame(pKF2);
        //判断是否是匹配点
        if(pMP1 && pMP2)
        {
            //进一步判断闭环帧是否观测到了pMP2
            if(!pMP1->isBad() && !pMP2->isBad() && i2>=0)
            {
                g2o::VertexSBAPointXYZ* vPoint1 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D1w = pMP1->GetWorldPos();
                //投影到相机坐标系
                cv::Mat P3D1c = R1w*P3D1w + t1w;
                //估值为相机坐标系下的坐标
                vPoint1->setEstimate(Converter::toVector3d(P3D1c));
                vPoint1->setId(id1);
                vPoint1->setFixed(true);
                optimizer.addVertex(vPoint1);

                g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D2w = pMP2->GetWorldPos();
                cv::Mat P3D2c = R2w*P3D2w + t2w;
                vPoint2->setEstimate(Converter::toVector3d(P3D2c));
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);
            }
            else
                continue;
        }
        else
            continue;
        //添加的边的数量
        nCorrespondences++;

        // SET EDGE x1 = S12*X2
        Eigen::Matrix<double,2,1> obs1;
        cv::KeyPoint kpUn1 = pKF1->GetKeyPointUn(i);
        obs1 << kpUn1.pt.x, kpUn1.pt.y;

        g2o::EdgeSim3ProjectXYZ* e12 = new g2o::EdgeSim3ProjectXYZ();
        //pMP2在闭环帧相机坐标系下的坐标
        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id2)));
        //sim3
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        //观测值为MapPoint在当前帧投影的像素坐标
        /*
        * void computeError()
        * {
        *   const VertexSim3Expmap* v1 = static_cast<const VertexSim3Expmap*>(_vertices[1]);
        *   const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
        *
        *   Vector2d obs(_measurement);
        *   _error = obs-v1->cam_map1(project(v1->estimate().map(v2->estimate())));
        * }
        */
        //e12的顶点:
        //1.v2 = pMP2在闭环帧相机坐标系下的投影
        //2.v1 = sim3(闭环帧变换到当前帧)
        //观测值为：当前帧观测到的MapPoint在图像上的投影
        //根据computeError可以看出，误差为pMP2在闭环帧相机坐标系下的坐标投影到当前帧图像上
        //与当前帧MapPoint对应的关键点像素坐标的差值
        e12->setMeasurement(obs1);
        float invSigmaSquare1 = pKF1->GetInvSigma2(kpUn1.octave);
        e12->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare1);

        g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);
        optimizer.addEdge(e12);

        // SET EDGE x2 = S21*X1
        //原理同上，误差为当前帧MapPoint在相机坐标系下的坐标投影到闭环帧图像上
        //与闭环帧MapPoint对应的关键点的像素坐标之差        
        Eigen::Matrix<double,2,1> obs2;
        cv::KeyPoint kpUn2 = pKF2->GetKeyPointUn(i2);
        obs2 << kpUn2.pt.x, kpUn2.pt.y;

        g2o::EdgeInverseSim3ProjectXYZ* e21 = new g2o::EdgeInverseSim3ProjectXYZ();

        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id1)));
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e21->setMeasurement(obs2);
        float invSigmaSquare2 = pKF2->GetSigma2(kpUn2.octave);
        e21->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare2);

        g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);
        optimizer.addEdge(e21);

        vpEdges12.push_back(e12);
        vpEdges21.push_back(e21);
        vnIndexEdge.push_back(i);
    }

    // Optimize

    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // Check inliers
    int nBad=0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=NULL;
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdges12[i]=NULL;
            vpEdges21[i]=NULL;
            nBad++;
        }
    }

    int nMoreIterations;
    if(nBad>0)
        nMoreIterations=10;
    else
        nMoreIterations=5;

    if(nCorrespondences-nBad<10)
        return 0;

    // Optimize again only with inliers

    optimizer.initializeOptimization();
    optimizer.optimize(nMoreIterations);

    int nIn = 0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=NULL;
        }
        else
            nIn++;
    }

    // Recover optimized Sim3
    g2o::VertexSim3Expmap* vSim3_recov = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
    //将优化后的sim3返回
    g2oS12= vSim3_recov->estimate();

    return nIn;
}

} //namespace ORB_SLAM
