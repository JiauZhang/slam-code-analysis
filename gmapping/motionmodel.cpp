#include "motionmodel.h"
#include <utils/stat.h>
#include <iostream>

#define MotionModelConditioningLinearCovariance 0.01
#define MotionModelConditioningAngularCovariance 0.001

namespace GMapping {



OrientedPoint 
MotionModel::drawFromMotion (const OrientedPoint& p, double linearMove, double angularMove) const{
	OrientedPoint n(p);
	double lm=linearMove  + fabs( linearMove ) * sampleGaussian( srr ) + fabs( angularMove ) * sampleGaussian( str );
	double am=angularMove + fabs( linearMove ) * sampleGaussian( srt ) + fabs( angularMove ) * sampleGaussian( stt );
	n.x+=lm*cos(n.theta+.5*am);
	n.y+=lm*sin(n.theta+.5*am);
	n.theta+=am;
	n.theta=atan2(sin(n.theta), cos(n.theta));
	return n;
}

OrientedPoint 
// p 为粒子位姿 https://blog.csdn.net/qq6689500/article/details/94641103
// 粒子表示的是机器人的世界坐标系，里程计表示的是机器人从其实点开始的坐标系下的坐标
// 当然两者可以重合
MotionModel::drawFromMotion(const OrientedPoint& p, const OrientedPoint& pnew, const OrientedPoint& pold) const{
	double sxy=0.3*srr;
	// 获得当前时刻位姿与上一时刻位姿相对变化在 里程计 下的表示
	OrientedPoint delta=absoluteDifference(pnew, pold);
	OrientedPoint noisypoint(delta);
	// 采样
	noisypoint.x+=sampleGaussian(srr*fabs(delta.x)+str*fabs(delta.theta)+sxy*fabs(delta.y));
	noisypoint.y+=sampleGaussian(srr*fabs(delta.y)+str*fabs(delta.theta)+sxy*fabs(delta.x));
	noisypoint.theta+=sampleGaussian(stt*fabs(delta.theta)+srt*sqrt(delta.x*delta.x+delta.y*delta.y));
	noisypoint.theta=fmod(noisypoint.theta, 2*M_PI);
	if (noisypoint.theta>M_PI)
		noisypoint.theta-=2*M_PI;
	// 最后的结果就是，根据里程计的两个位姿得到相对位姿
	// 然后转换到世界坐标系下的增量，加噪声之后转化成世界坐标系下粒子下一时刻的状态
	return absoluteSum(p,noisypoint);
}


/*
OrientedPoint 
MotionModel::drawFromMotion(const OrientedPoint& p, const OrientedPoint& pnew, const OrientedPoint& pold) const{
	
	//compute the three stps needed for perfectly matching the two poses if the noise is absent
	
	OrientedPoint delta=pnew-pold;
	double aoffset=atan2(delta.y, delta.x);
	double alpha1=aoffset-pold.theta;
	alpha1=atan2(sin(alpha1), cos(alpha1));
	double rho=sqrt(delta*delta);
	double alpha2=pnew.theta-aoffset;
	alpha2=atan2(sin(alpha2), cos(alpha2));
	
	OrientedPoint pret=drawFromMotion(p, 0, alpha1);
	pret=drawFromMotion(pret, rho, 0);
	pret=drawFromMotion(pret, 0, alpha2);
	return pret;
}
*/


Covariance3 MotionModel::gaussianApproximation(const OrientedPoint& pnew, const OrientedPoint& pold) const{
	OrientedPoint delta=absoluteDifference(pnew,pold);
	double linearMove=sqrt(delta.x*delta.x+delta.y*delta.y);
	double angularMove=fabs(delta.x);
	double s11=srr*srr*linearMove*linearMove;
	double s22=stt*stt*angularMove*angularMove;
	double s12=str*angularMove*srt*linearMove;
	Covariance3 cov;
	double s=sin(pold.theta),c=cos(pold.theta);
	cov.xx=c*c*s11+MotionModelConditioningLinearCovariance;
	cov.yy=s*s*s11+MotionModelConditioningLinearCovariance;
	cov.tt=s22+MotionModelConditioningAngularCovariance;
	cov.xy=s*c*s11;
	cov.xt=c*s12;
	cov.yt=s*s12;
	return cov;
}

};

