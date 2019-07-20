#ifndef SCANMATCHER_H
#define SCANMATCHER_H

#include "icp.h"
#include "smmap.h"
#include <utils/macro_params.h>
#include <utils/stat.h>
#include <iostream>
#include <utils/gvalues.h>
#define LASER_MAXBEAMS 1024

namespace GMapping {

class ScanMatcher{
	public:
		typedef Covariance3 CovarianceMatrix;
		
		ScanMatcher();
		double icpOptimize(OrientedPoint& pnew, const ScanMatcherMap& map, const OrientedPoint& p, const double* readings) const;
		double optimize(OrientedPoint& pnew, const ScanMatcherMap& map, const OrientedPoint& p, const double* readings) const;
		double optimize(OrientedPoint& mean, CovarianceMatrix& cov, const ScanMatcherMap& map, const OrientedPoint& p, const double* readings) const;
		
		double   registerScan(ScanMatcherMap& map, const OrientedPoint& p, const double* readings);
		void setLaserParameters
			(unsigned int beams, double* angles, const OrientedPoint& lpose);
		void setMatchingParameters
			(double urange, double range, double sigma, int kernsize, double lopt, double aopt, int iterations, double likelihoodSigma=1, unsigned int likelihoodSkip=0 );
		void invalidateActiveArea();
		void computeActiveArea(ScanMatcherMap& map, const OrientedPoint& p, const double* readings);

		inline double icpStep(OrientedPoint & pret, const ScanMatcherMap& map, const OrientedPoint& p, const double* readings) const;
		inline double score(const ScanMatcherMap& map, const OrientedPoint& p, const double* readings) const;
		inline unsigned int likelihoodAndScore(double& s, double& l, const ScanMatcherMap& map, const OrientedPoint& p, const double* readings) const;
		double likelihood(double& lmax, OrientedPoint& mean, CovarianceMatrix& cov, const ScanMatcherMap& map, const OrientedPoint& p, const double* readings);
		double likelihood(double& _lmax, OrientedPoint& _mean, CovarianceMatrix& _cov, const ScanMatcherMap& map, const OrientedPoint& p, Gaussian3& odometry, const double* readings, double gain=180.);
		inline const double* laserAngles() const { return m_laserAngles; }
		inline unsigned int laserBeams() const { return m_laserBeams; }
		
		static const double nullLikelihood;
	protected:
		//state of the matcher
		bool m_activeAreaComputed;
		
		/**laser parameters*/
		unsigned int m_laserBeams;
		double       m_laserAngles[LASER_MAXBEAMS];
		//OrientedPoint m_laserPose;
		PARAM_SET_GET(OrientedPoint, laserPose, protected, public, public)
		PARAM_SET_GET(double, laserMaxRange, protected, public, public)
		/**scan_matcher parameters*/
		PARAM_SET_GET(double, usableRange, protected, public, public)
		PARAM_SET_GET(double, gaussianSigma, protected, public, public)
		PARAM_SET_GET(double, likelihoodSigma, protected, public, public)
		PARAM_SET_GET(int,    kernelSize, protected, public, public)
		PARAM_SET_GET(double, optAngularDelta, protected, public, public)
		PARAM_SET_GET(double, optLinearDelta, protected, public, public)
		PARAM_SET_GET(unsigned int, optRecursiveIterations, protected, public, public)
		PARAM_SET_GET(unsigned int, likelihoodSkip, protected, public, public)
		PARAM_SET_GET(double, llsamplerange, protected, public, public)
		PARAM_SET_GET(double, llsamplestep, protected, public, public)
		PARAM_SET_GET(double, lasamplerange, protected, public, public)
		PARAM_SET_GET(double, lasamplestep, protected, public, public)
		PARAM_SET_GET(bool, generateMap, protected, public, public)
		PARAM_SET_GET(double, enlargeStep, protected, public, public)
		PARAM_SET_GET(double, fullnessThreshold, protected, public, public)
		PARAM_SET_GET(double, angularOdometryReliability, protected, public, public)
		PARAM_SET_GET(double, linearOdometryReliability, protected, public, public)
		PARAM_SET_GET(double, freeCellRatio, protected, public, public)
		PARAM_SET_GET(unsigned int, initialBeamsSkip, protected, public, public)
};

inline double ScanMatcher::icpStep(OrientedPoint & pret, const ScanMatcherMap& map, const OrientedPoint& p, const double* readings) const{
	const double * angle=m_laserAngles+m_initialBeamsSkip;
	OrientedPoint lp=p;
	lp.x+=cos(p.theta)*m_laserPose.x-sin(p.theta)*m_laserPose.y;
	lp.y+=sin(p.theta)*m_laserPose.x+cos(p.theta)*m_laserPose.y;
	lp.theta+=m_laserPose.theta;
	unsigned int skip=0;
	double freeDelta=map.getDelta()*m_freeCellRatio;
	std::list<PointPair> pairs;
	
	for (const double* r=readings+m_initialBeamsSkip; r<readings+m_laserBeams; r++, angle++){
		skip++;
		skip=skip>m_likelihoodSkip?0:skip;
		if (*r>m_usableRange) continue;
		if (skip) continue;
		Point phit=lp;
		phit.x+=*r*cos(lp.theta+*angle);
		phit.y+=*r*sin(lp.theta+*angle);
		IntPoint iphit=map.world2map(phit);
		Point pfree=lp;
		pfree.x+=(*r-map.getDelta()*freeDelta)*cos(lp.theta+*angle);
		pfree.y+=(*r-map.getDelta()*freeDelta)*sin(lp.theta+*angle);
 		pfree=pfree-phit;
		IntPoint ipfree=map.world2map(pfree);
		bool found=false;
		Point bestMu(0.,0.);
		Point bestCell(0.,0.);
		for (int xx=-m_kernelSize; xx<=m_kernelSize; xx++)
		for (int yy=-m_kernelSize; yy<=m_kernelSize; yy++){
			IntPoint pr=iphit+IntPoint(xx,yy);
			IntPoint pf=pr+ipfree;
			//AccessibilityState s=map.storage().cellState(pr);
			//if (s&Inside && s&Allocated){
				const PointAccumulator& cell=map.cell(pr);
				const PointAccumulator& fcell=map.cell(pf);
				if (((double)cell )> m_fullnessThreshold && ((double)fcell )<m_fullnessThreshold){
					Point mu=phit-cell.mean();
					if (!found){
						bestMu=mu;
						bestCell=cell.mean();
						found=true;
					}else
						if((mu*mu)<(bestMu*bestMu)){
							bestMu=mu;
							bestCell=cell.mean();
						} 
						
				}
			//}
		}
		if (found){
			pairs.push_back(std::make_pair(phit, bestCell));
			//std::cerr << "(" << phit.x-bestCell.x << "," << phit.y-bestCell.y << ") ";
		}
		//std::cerr << std::endl;
	}
	
	OrientedPoint result(0,0,0);
	//double icpError=icpNonlinearStep(result,pairs);
	std::cerr << "result(" << pairs.size() << ")=" << result.x << " " << result.y << " " << result.theta << std::endl;
	pret.x=p.x+result.x;
	pret.y=p.y+result.y;
	pret.theta=p.theta+result.theta;
	pret.theta=atan2(sin(pret.theta), cos(pret.theta));
	return score(map, p, readings);
}
// 根据粒子位姿，遍历激光束测距数据，计算障碍物累计得分
inline double ScanMatcher::score(const ScanMatcherMap& map, const OrientedPoint& p, const double* readings) const{
	double s=0;
	const double * angle=m_laserAngles+m_initialBeamsSkip;
	OrientedPoint lp=p;
	// 得到 laser 的位姿
	lp.x+=cos(p.theta)*m_laserPose.x-sin(p.theta)*m_laserPose.y;
	lp.y+=sin(p.theta)*m_laserPose.x+cos(p.theta)*m_laserPose.y;
	lp.theta+=m_laserPose.theta;
	unsigned int skip=0;
	double freeDelta=map.getDelta()*m_freeCellRatio;
	// 遍历测距数据
	for (const double* r=readings+m_initialBeamsSkip; r<readings+m_laserBeams; r++, angle++){
		skip++;
		// 并不是所有的 测距束 都使用，而是均匀跳过一些
		skip=skip>m_likelihoodSkip?0:skip;
		// 超过可用范围的也不用
		if (*r>m_usableRange) continue;
		if (skip) continue;
		Point phit=lp;
		// 障碍物位置
		phit.x+=*r*cos(lp.theta+*angle);
		phit.y+=*r*sin(lp.theta+*angle);
		// 障碍物在 map 中的坐标
		IntPoint iphit=map.world2map(phit);
		Point pfree=lp;
		pfree.x+=(*r-map.getDelta()*freeDelta)*cos(lp.theta+*angle);
		pfree.y+=(*r-map.getDelta()*freeDelta)*sin(lp.theta+*angle);
		// pfree 表示 最后一个 free 栅格 与 占用栅格 的坐标 差值
 		pfree=pfree-phit;
		// laser 到障碍物前方认为是        	free 的终点在 map 中的坐标
		// 最后一个 free 栅格 与 占用栅格 的坐标 差值 在 map 中的表示
		IntPoint ipfree=map.world2map(pfree);
		bool found=false;
		Point bestMu(0.,0.);
		// 对障碍物周围 -m_kernelSize 到 +m_kernelSize 的正方形区域内都更新权重
		for (int xx=-m_kernelSize; xx<=m_kernelSize; xx++)
		for (int yy=-m_kernelSize; yy<=m_kernelSize; yy++){
			// 方形区域点的坐标
			IntPoint pr=iphit+IntPoint(xx,yy);
			// 认为是 free 的点坐标
			IntPoint pf=pr+ipfree;
			//AccessibilityState s=map.storage().cellState(pr);
			//if (s&Inside && s&Allocated){
				const PointAccumulator& cell=map.cell(pr);
				const PointAccumulator& fcell=map.cell(pf);
				// 假设phit是被激光击中的点，这样的话沿着激光方向的前面一个点必定的空闲的
				// 故检查占用概率是否满足要求
				// inline operator double() const { return visits?(double)n*SIGHT_INC/(double)visits:-1; }
				// 强制类型转化定义为返回占用概率
				if (((double)cell )> m_fullnessThreshold && ((double)fcell )<m_fullnessThreshold){
					// 获取当前扫描击中的坐标 与 该栅格被占用的均值坐标 的差值
					// 差值越小评分越高
					Point mu=phit-cell.mean();
					// 选出最可能是障碍物的点
					if (!found){
						bestMu=mu;
						found=true;
					}else
						bestMu=(mu*mu)<(bestMu*bestMu)?mu:bestMu;
				}
			//}
		}
		// 累计得分
		if (found)
			s+=exp(-1./m_gaussianSigma*bestMu*bestMu);
	}
	return s;
}
// 根据粒子位姿计算得分和似然
inline unsigned int ScanMatcher::likelihoodAndScore(double& s, double& l, const ScanMatcherMap& map, const OrientedPoint& p, const double* readings) const{
	using namespace std;
	l=0;
	s=0;
	// 激光束角度
	const double * angle=m_laserAngles+m_initialBeamsSkip;
	// 粒子位姿
	OrientedPoint lp=p;
	// 激光的位姿
	lp.x+=cos(p.theta)*m_laserPose.x-sin(p.theta)*m_laserPose.y;
	lp.y+=sin(p.theta)*m_laserPose.x+cos(p.theta)*m_laserPose.y;
	lp.theta+=m_laserPose.theta;
	double noHit=nullLikelihood/(m_likelihoodSigma);
	unsigned int skip=0;
	unsigned int c=0;
	double freeDelta=map.getDelta()*m_freeCellRatio;
	// 遍历激光束测距数据
	for (const double* r=readings+m_initialBeamsSkip; r<readings+m_laserBeams; r++, angle++){
		skip++;
		// 同样是等间隔跳过一些测距数据
		skip=skip>m_likelihoodSkip?0:skip;
		if (*r>m_usableRange) continue;
		if (skip) continue;
		Point phit=lp;
		phit.x+=*r*cos(lp.theta+*angle);
		phit.y+=*r*sin(lp.theta+*angle);
		// 障碍物坐标
		IntPoint iphit=map.world2map(phit);
		Point pfree=lp;
		pfree.x+=(*r-freeDelta)*cos(lp.theta+*angle);
		pfree.y+=(*r-freeDelta)*sin(lp.theta+*angle);
		// 障碍物与 free 的栅格的距离
		pfree=pfree-phit;
		// 这个距离在 map 中的表示
		IntPoint ipfree=map.world2map(pfree);
		bool found=false;
		Point bestMu(0.,0.);
		// 遍历障碍物周围 m_kernelSize 的栅格
		for (int xx=-m_kernelSize; xx<=m_kernelSize; xx++)
		for (int yy=-m_kernelSize; yy<=m_kernelSize; yy++){
			IntPoint pr=iphit+IntPoint(xx,yy);
			IntPoint pf=pr+ipfree;
			//AccessibilityState s=map.storage().cellState(pr);
			//if (s&Inside && s&Allocated){
				const PointAccumulator& cell=map.cell(pr);
				const PointAccumulator& fcell=map.cell(pf);
				// 判断是否可能是障碍物
				if (((double)cell )>m_fullnessThreshold && ((double)fcell )<m_fullnessThreshold){
					Point mu=phit-cell.mean();
					if (!found){
						bestMu=mu;
						found=true;
					}else
						bestMu=(mu*mu)<(bestMu*bestMu)?mu:bestMu;
				}
			//}	
		}
		// 累计得分
		if (found){
			s+=exp(-1./m_gaussianSigma*bestMu*bestMu);
			c++;
		}
		if (!skip){
			// 似然的指数部分
			double f=(-1./m_likelihoodSigma)*(bestMu*bestMu);
			l+=(found)?f:noHit;
		}
	}
	return c;
}

};

#endif
