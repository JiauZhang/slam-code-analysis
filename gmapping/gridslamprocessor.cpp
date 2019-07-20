
#include <string>
#include <deque>
#include <list>
#include <map>
#include <set>
#include <fstream>
#include <iomanip>
#include <utils/stat.h>
#include "gridslamprocessor.h"

//#define MAP_CONSISTENCY_CHECK
//#define GENERATE_TRAJECTORIES
namespace GMapping
{

	const double	m_distanceThresholdCheck = 20;

	using namespace std;

	GridSlamProcessor::GridSlamProcessor(): m_infoStream(cout) {

		m_obsSigmaGain		= 1;
		m_resampleThreshold = 0.5;
		m_minimumScore		= 0.;
	}

	GridSlamProcessor::GridSlamProcessor(const GridSlamProcessor & gsp): m_particles(gsp.m_particles), 
	m_infoStream(cout) {

		m_obsSigmaGain		= gsp.m_obsSigmaGain;
		m_resampleThreshold = gsp.m_resampleThreshold;
		m_minimumScore		= gsp.m_minimumScore;

		m_beams 			= gsp.m_beams;
		m_indexes			= gsp.m_indexes;
		m_motionModel		= gsp.m_motionModel;
		m_resampleThreshold = gsp.m_resampleThreshold;
		m_matcher			= gsp.m_matcher;

		m_count 			= gsp.m_count;
		m_readingCount		= gsp.m_readingCount;
		m_lastPartPose		= gsp.m_lastPartPose;
		m_pose				= gsp.m_pose;
		m_odoPose			= gsp.m_odoPose;
		m_linearDistance	= gsp.m_linearDistance;
		m_angularDistance	= gsp.m_angularDistance;
		m_neff				= gsp.m_neff;

		cerr << "FILTER COPY CONSTRUCTOR" << endl;
		cerr << "m_odoPose=" << m_odoPose.x << " " << m_odoPose.y << " " << m_odoPose.theta << endl;
		cerr << "m_lastPartPose=" << m_lastPartPose.x << " " << m_lastPartPose.y << " " << m_lastPartPose.theta << endl;
		cerr << "m_linearDistance=" << m_linearDistance << endl;
		cerr << "m_angularDistance=" << m_linearDistance << endl;


		m_xmin				= gsp.m_xmin;
		m_ymin				= gsp.m_ymin;
		m_xmax				= gsp.m_xmax;
		m_ymax				= gsp.m_ymax;
		m_delta 			= gsp.m_delta;

		m_regScore			= gsp.m_regScore;
		m_critScore 		= gsp.m_critScore;
		m_maxMove			= gsp.m_maxMove;

		m_linearThresholdDistance = gsp.m_linearThresholdDistance;
		m_angularThresholdDistance = gsp.m_angularThresholdDistance;
		m_obsSigmaGain		= gsp.m_obsSigmaGain;

#ifdef MAP_CONSISTENCY_CHECK
		cerr << __PRETTY_FUNCTION__ << ": trajectories copy.... ";
#endif

		TNodeVector 	v	= gsp.getTrajectories();

		for (unsigned int i = 0; i < v.size(); i++) {
			m_particles[i].node = v[i];
		}

#ifdef MAP_CONSISTENCY_CHECK
		cerr << "end" << endl;
#endif


		cerr << "Tree: normalizing, resetting and propagating weights within copy construction/cloneing ...";
		updateTreeWeights(false);
		cerr << ".done!" << endl;
	}

	GridSlamProcessor::GridSlamProcessor(std::ostream & infoS): m_infoStream(infoS) {
		m_obsSigmaGain		= 1;
		m_resampleThreshold = 0.5;
		m_minimumScore		= 0.;

	}

	GridSlamProcessor * GridSlamProcessor::clone()

	const {
#ifdef MAP_CONSISTENCY_CHECK
		cerr << __PRETTY_FUNCTION__ << ": performing preclone_fit_test" << endl;
		typedef std::map < autoptr < Array2D < PointAccumulator > >::reference * const, int > PointerMap;
		PointerMap		pmap;

		for (ParticleVector::const_iterator it = m_particles.begin(); it != m_particles.end(); it++) {
			const ScanMatcherMap & m1(it->map);
			const HierarchicalArray2D < PointAccumulator > &h1(m1.storage());

			for (int x = 0; x < h1.getXSize(); x++) {
				for (int y = 0; y < h1.getYSize(); y++) {
					const autoptr < Array2D < PointAccumulator > > &a1(h1.m_cells[x][y]);

					if (a1.m_reference) {
						PointerMap::iterator f = pmap.find(a1.m_reference);

						if (f == pmap.end())
							pmap.insert(make_pair(a1.m_reference, 1));
						else 
							f->second++;
					}
				}
			}
		}

		cerr << __PRETTY_FUNCTION__ << ": Number of allocated chunks" << pmap.size() << endl;

		for (PointerMap::const_iterator it = pmap.begin(); it != pmap.end(); it++)
			assert(it->first->shares == (unsigned int) it->second);

		cerr << __PRETTY_FUNCTION__ << ": SUCCESS, the error is somewhere else" << endl;
#endif

		GridSlamProcessor * cloned = new GridSlamProcessor(*this);

#ifdef MAP_CONSISTENCY_CHECK
		cerr << __PRETTY_FUNCTION__ << ": trajectories end" << endl;
		cerr << __PRETTY_FUNCTION__ << ": performing afterclone_fit_test" << endl;
		ParticleVector::const_iterator jt = cloned->m_particles.begin();

		for (ParticleVector::const_iterator it = m_particles.begin(); it != m_particles.end(); it++) {
			const ScanMatcherMap & m1(it->map);
			const ScanMatcherMap & m2(jt->map);
			const HierarchicalArray2D < PointAccumulator > &h1(m1.storage());
			const HierarchicalArray2D < PointAccumulator > &h2(m2.storage());

			jt++;

			for (int x = 0; x < h1.getXSize(); x++) {
				for (int y = 0; y < h1.getYSize(); y++) {
					const autoptr < Array2D < PointAccumulator > > &a1(h1.m_cells[x][y]);
					const autoptr < Array2D < PointAccumulator > > &a2(h2.m_cells[x][y]);

					assert(a1.m_reference == a2.m_reference);
					assert((!a1.m_reference) || ! (a1.m_reference->shares % 2));
				}
			}
		}

		cerr << __PRETTY_FUNCTION__ << ": SUCCESS, the error is somewhere else" << endl;
#endif

		return cloned;
	}

	GridSlamProcessor::~GridSlamProcessor() {
		cerr << __PRETTY_FUNCTION__ << ": Start" << endl;
		cerr << __PRETTY_FUNCTION__ << ": Deeting tree" << endl;

		for (std::vector < Particle >::iterator it = m_particles.begin(); it != m_particles.end(); it++) {
#ifdef TREE_CONSISTENCY_CHECK
			TNode * 		node = it->node;

			while (node)
				node = node->parent;

			cerr << "@" << endl;
#endif

			if (it->node)
				delete it->node;

			//cout << "l=" << it->weight<< endl;
		}

#ifdef MAP_CONSISTENCY_CHECK
		cerr << __PRETTY_FUNCTION__ << ": performing predestruction_fit_test" << endl;
		typedef std::map < autoptr < Array2D < PointAccumulator > >::reference * const, int > PointerMap;
		PointerMap		pmap;

		for (ParticleVector::const_iterator it = m_particles.begin(); it != m_particles.end(); it++) {
			const ScanMatcherMap & m1(it->map);
			const HierarchicalArray2D < PointAccumulator > &h1(m1.storage());

			for (int x = 0; x < h1.getXSize(); x++) {
				for (int y = 0; y < h1.getYSize(); y++) {
					const autoptr < Array2D < PointAccumulator > > &a1(h1.m_cells[x][y]);

					if (a1.m_reference) {
						PointerMap::iterator f = pmap.find(a1.m_reference);

						if (f == pmap.end())
							pmap.insert(make_pair(a1.m_reference, 1));
						else 
							f->second++;
					}
				}
			}
		}

		cerr << __PRETTY_FUNCTION__ << ": Number of allocated chunks" << pmap.size() << endl;

		for (PointerMap::const_iterator it = pmap.begin(); it != pmap.end(); it++)
			assert(it->first->shares >= (unsigned int) it->second);

		cerr << __PRETTY_FUNCTION__ << ": SUCCESS, the error is somewhere else" << endl;
#endif
	}



	void GridSlamProcessor::setMatchingParameters(double urange, double range, double sigma, int kernsize, 
		double		lopt, double aopt, 
		int 		iterations, double likelihoodSigma, double likelihoodGain, unsigned int likelihoodSkip) {
		m_obsSigmaGain		= likelihoodGain;
		m_matcher.setMatchingParameters(urange, range, sigma, kernsize, lopt, aopt, iterations, likelihoodSigma, 
			likelihoodSkip);

		if (m_infoStream)
			m_infoStream << " -maxUrange " << urange << " -maxUrange " << range << " -sigma 	" << sigma << " -kernelSize " << kernsize << " -lstep " << lopt << " -lobsGain " << m_obsSigmaGain << " -astep " << aopt << endl;


	}

	void GridSlamProcessor::setMotionModelParameters(double srr, double srt, double str, double stt) {
		m_motionModel.srr	= srr;
		m_motionModel.srt	= srt;
		m_motionModel.str	= str;
		m_motionModel.stt	= stt;

		if (m_infoStream)
			m_infoStream << " -srr " << srr << " -srt " << srt << " -str " << str << " -stt " << stt << endl;

	}

	void GridSlamProcessor::setUpdateDistances(double linear, double angular, double resampleThreshold) {
		m_linearThresholdDistance = linear;
		m_angularThresholdDistance = angular;
		m_resampleThreshold = resampleThreshold;

		if (m_infoStream)
			m_infoStream << " -linearUpdate " << linear << " -angularUpdate " << angular << " -resampleThreshold " << m_resampleThreshold << endl;
	}

	//HERE STARTS THE BEEF
	GridSlamProcessor::Particle::Particle(const ScanMatcherMap & m): map(m), pose(0, 0, 0), weight(0), 
	weightSum(0), 
	gweight(0), previousIndex(0) {
		node				= 0;
	}

	// 设置绑定的激光雷达传感器测距参数及初始位姿
	void GridSlamProcessor::setSensorMap(const SensorMap & smap) {

		/*
			Construct the angle table for the sensor
			
			FIXME For now detect the readings of only the front laser, and assume its pose is in the center of the 
			robot
			 
		*/
		SensorMap::const_iterator laser_it = smap.find(std::string("FLASER"));

		if (laser_it == smap.end()) {
			cerr << "Attempting to load the new carmen log format" << endl;
			laser_it			= smap.find(std::string("ROBOTLASER1"));
			assert(laser_it != smap.end());
		}

		const RangeSensor * rangeSensor = dynamic_cast < const RangeSensor * > ((laser_it->second));

		assert(rangeSensor && rangeSensor->beams().size());

		m_beams 			= static_cast < unsigned int > (rangeSensor->beams().size());
		double *		angles = new double[rangeSensor->beams().size()];

		for (unsigned int i = 0; i < m_beams; i++) {
			angles[i]			= rangeSensor->beams()[i].pose.theta;
		}

		m_matcher.setLaserParameters(m_beams, angles, rangeSensor->getPose());
		delete[] angles;
	}

	void GridSlamProcessor::init(unsigned int size, double xmin, double ymin, double xmax, double ymax, double delta, 
		OrientedPoint initialPose) {
		m_xmin				= xmin;
		m_ymin				= ymin;
		m_xmax				= xmax;
		m_ymax				= ymax;
		m_delta 			= delta;

		if (m_infoStream)
			m_infoStream << " -xmin " << m_xmin << " -xmax " << m_xmax << " -ymin " << m_ymin << " -ymax " << m_ymax << " -delta " << m_delta << " -particles " << size << endl;


		m_particles.clear();
		TNode * 		node = new TNode(initialPose, 0, 0, 0);		
		/*
			Map<PointAccumulator,HierarchicalArray2D<PointAccumulator> > ScanMatcherMap;
			template <class Cell, class Storage, const bool isClass=true> 
			对比Map的模板参数句可以知道
			cell = PointAccumulator, Storage = HierarchicalArray2D<PointAccumulator>
			template <class Cell>
			class HierarchicalArray2D: public Array2D<autoptr< Array2D<Cell> > >
			一切都明了了 ~~~!!!!
		*/
		// 关键是这地图是局部变量呀，用完就释放了？
		ScanMatcherMap lmap(Point(xmin + xmax, ymin + ymax) *.5, xmax - xmin, ymax - ymin, delta);
		// 初始化 粒子
		for (unsigned int i = 0; i < size; i++) {
			// 每个粒子都绑定地图
			m_particles.push_back(Particle(lmap));
			m_particles.back().pose = initialPose;
			m_particles.back().previousPose = initialPose;
			m_particles.back().setWeight(0);
			m_particles.back().previousIndex = 0;

			// this is not needed
			//		m_particles.back().node=new TNode(initialPose, 0, node, 0);
			// we use the root directly
			// 绑定第一个轨迹节点
			m_particles.back().node = node;
		}
		// m_neff = 粒子数量
		m_neff				= (double)
		size;
		m_count 			= 0;
		m_readingCount		= 0;
		m_linearDistance	= m_angularDistance = 0;
	}

	void GridSlamProcessor::processTruePos(const OdometryReading & o) {
		const OdometrySensor * os = dynamic_cast < const OdometrySensor * > (o.getSensor());

		if (os && os->isIdeal() && m_outputStream) {
			m_outputStream << setiosflags(ios::fixed) << setprecision(3);
			m_outputStream << "SIMULATOR_POS " << o.getPose().x << " " << o.getPose().y << " ";
			m_outputStream << setiosflags(ios::fixed) << setprecision(6) << o.getPose().theta << " " << o.getTime() << endl;
		}
	}

	// adaptParticles默认 = 0
	// 返回值标识当前帧扫描数据是否使用了
	bool GridSlamProcessor::processScan(const RangeReading & reading, int adaptParticles) {

		/**retireve the position from the reading, and compute the odometry*/
	// 取出扫描数据中记录的里程计位姿
		OrientedPoint	relPose = reading.getPose();
	// GridSlamProcessor::init 中将 m_count 初始化为 0
	// m_odoPose 实际用作记录上一时刻的位姿，由于是第一次循环，故直接等于0时刻位姿
		if (!m_count) {
			m_lastPartPose		= m_odoPose = relPose;
		}

		//write the state of the reading and update all the particles using the motion model
		for (ParticleVector::iterator it = m_particles.begin(); it != m_particles.end(); it++) {
			OrientedPoint & pose(it->pose);
			// 粒子新的位姿，relPose 当前时刻位姿，也是激光中心的位姿， m_odoPose 上一时刻位姿
			pose				= m_motionModel.drawFromMotion(it->pose, relPose, m_odoPose);
		}

		// update the output file
		if (m_outputStream.is_open()) {
			m_outputStream << setiosflags(ios::fixed) << setprecision(6);
			m_outputStream << "ODOM ";
			m_outputStream << setiosflags(ios::fixed) << setprecision(3) << m_odoPose.x << " " << m_odoPose.y << " ";
			m_outputStream << setiosflags(ios::fixed) << setprecision(6) << m_odoPose.theta << " ";
			m_outputStream << reading.getTime();
			m_outputStream << endl;
		}

		if (m_outputStream.is_open()) {
			m_outputStream << setiosflags(ios::fixed) << setprecision(6);
			m_outputStream << "ODO_UPDATE " << m_particles.size() << " ";

			for (ParticleVector::iterator it = m_particles.begin(); it != m_particles.end(); it++) {
				OrientedPoint & pose(it->pose);

				m_outputStream << setiosflags(ios::fixed) << setprecision(3) << pose.x << " " << pose.y << " ";
				m_outputStream << setiosflags(ios::fixed) << setprecision(6) << pose.theta << " " << it->weight << " ";
			}

			m_outputStream << reading.getTime();
			m_outputStream << endl;
		}

		//invoke the callback
		// 空函数，不明觉厉~
		onOdometryUpdate();


		// accumulate the robot translation and rotation
		// 上一时刻位姿与当前时刻位姿的相对运动
		OrientedPoint	move = relPose - m_odoPose;

		move.theta			= atan2(sin(move.theta), cos(move.theta));
		// 累计旋转和平移
		m_linearDistance	+= sqrt(move * move);
		m_angularDistance	+= fabs(move.theta);

		// if the robot jumps throw a warning
		if (m_linearDistance > m_distanceThresholdCheck) {
			cerr << "***********************************************************************" << endl;
			cerr << "********** Error: m_distanceThresholdCheck overridden!!!! *************" << endl;
			cerr << "m_distanceThresholdCheck=" << m_distanceThresholdCheck << endl;
			cerr << "Old Odometry Pose= " << m_odoPose.x << " " << m_odoPose.y << " " << m_odoPose.theta << endl;
			cerr << "New Odometry Pose (reported from observation)= " << relPose.x << " " << relPose.y << " " << relPose.theta << endl;
			cerr << "***********************************************************************" << endl;
			cerr << "** The Odometry has a big jump here. This is probably a bug in the 	**" << endl;
			cerr << "** odometry/laser input. We continue now, but the result is probably **" << endl;
			cerr << "** crap or can lead to a core dump since the map doesn't fit.... C&G **" << endl;
			cerr << "***********************************************************************" << endl;
		}
		// 原来 m_odoPose 是记录当前时刻的位姿 relPose，以备下次循环使用
		m_odoPose			= relPose;

		bool			processed = false;

		// process a scan only if the robot has traveled a given distance
		// 第一次 肯定要执行该分支
		// 后续帧只在 平移距离够远、旋转角够大时才执行，控制执行频率
		// m_*记录的是绝对值的和，即它记录的变化的和，并不是真正的平移了多远、或者旋转了多大角
		// 而是变化的动作是否够多了
		if (!m_count || m_linearDistance > m_linearThresholdDistance ||
			 m_angularDistance > m_angularThresholdDistance) {

			if (m_outputStream.is_open()) {
				m_outputStream << setiosflags(ios::fixed) << setprecision(6);
				m_outputStream << "FRAME " << m_readingCount;
				m_outputStream << " " << m_linearDistance;
				m_outputStream << " " << m_angularDistance << endl;
			}

			if (m_infoStream)
				m_infoStream << "update frame " << m_readingCount << endl << "update ld=" << m_linearDistance << " ad=" << m_angularDistance << endl;


			cerr << "Laser Pose= " << reading.getPose().x << " " << reading.getPose().y << " " << reading.getPose().theta << endl;


			//this is for converting the reading in a scan-matcher feedable form
			assert(reading.size() == m_beams);
			double *		plainReading = new double[m_beams];
			// 取出当前帧激光测距数据
			for (unsigned int i = 0; i < m_beams; i++) {
				plainReading[i] 	= reading[i];
			}

			m_infoStream << "m_count " << m_count << endl;
			// 第一次循环不会执行这个分支，因为现在只有一帧激光数据，没法进行 scanMatch
			// 后续帧执行这里
			if (m_count > 0) {
				// 方法实现在：gridslamprocessor.hxx 中
				// 遍历粒子集，通过对初始位姿添加扰动，然后根据得分得到最优的位姿，然后更新粒子权重
				// 然后计算在最优位姿下的活跃栅格点
				scanMatch(plainReading);

				if (m_outputStream.is_open()) {
					m_outputStream << "LASER_READING " << reading.size() << " ";
					m_outputStream << setiosflags(ios::fixed) << setprecision(2);

					for (RangeReading::const_iterator b = reading.begin(); b != reading.end(); b++) {
						m_outputStream << *b << " ";
					}

					OrientedPoint	p	= reading.getPose();

					m_outputStream << setiosflags(ios::fixed) << setprecision(6);
					m_outputStream << p.x << " " << p.y << " " << p.theta << " " << reading.getTime() << endl;
					m_outputStream << "SM_UPDATE " << m_particles.size() << " ";

					for (ParticleVector::const_iterator it = m_particles.begin(); it != m_particles.end(); it++) {
						const OrientedPoint & pose = it->pose;

						m_outputStream << setiosflags(ios::fixed) << setprecision(3) << pose.x << " " << pose.y << " ";
						m_outputStream << setiosflags(ios::fixed) << setprecision(6) << pose.theta << " " << it->weight << " ";
					}

					m_outputStream << endl;
				}
				// 又是空？
				onScanmatchUpdate();

				updateTreeWeights(false);

				if (m_infoStream) {
					m_infoStream << "neff= " << m_neff << endl;
				}

				if (m_outputStream.is_open()) {
					m_outputStream << setiosflags(ios::fixed) << setprecision(6);
					m_outputStream << "NEFF " << m_neff << endl;
				}
				// adaptParticles 为 0
				// bool resample(const double* plainReading, int adaptParticles, 
			 	// const RangeReading* rr=0)
			 	// 在 gridslamprocessor.hxx 中实现
				resample(plainReading, adaptParticles);

			}
			// 第一次执行该分支
			else {
				m_infoStream << "Registering First Scan" << endl;

				for (ParticleVector::iterator it = m_particles.begin(); it != m_particles.end(); it++) {
					// 设置需要更新活跃区域
					m_matcher.invalidateActiveArea();
					// 更新机器人位置和各个障碍物之间连线所覆盖的map中的栅格，及障碍物所在栅格状态
					m_matcher.computeActiveArea(it->map, it->pose, plainReading);
					m_matcher.registerScan(it->map, it->pose, plainReading);

					// cyr: not needed anymore, particles refer to the root in the beginning!
					// 输入参数：粒子位姿、权重、父节点、孩子数量
					// 此时的 it->node == NULL, 所以第一个节点的 parent = NULL
					TNode * 		node = new TNode(it->pose, 0., it->node, 0);
					// 需要关联的激光数据
					node->reading		= 0;
					// 粒子绑定该节点
					it->node			= node;

				}
			}

			//		cerr	<< "Tree: normalizing, resetting and propagating weights at the end..." ;
			updateTreeWeights(false);

			//		cerr	<< ".done!" <<endl;
			// 删除激光扫描数据
			delete[] plainReading;
			m_lastPartPose		= m_odoPose;		//update the past pose for the next iteration
			m_linearDistance	= 0;
			m_angularDistance	= 0;
			// 该分支每进来一次 m_count++
			m_count++;
			processed			= true;

			//keep ready for the next step
			// 迭代当前位姿状态
			for (ParticleVector::iterator it = m_particles.begin(); it != m_particles.end(); it++) {
				it->previousPose	= it->pose;
			}

		}

		if (m_outputStream.is_open())
			m_outputStream << flush;
		// 记录总的 scan 数量
		m_readingCount++;
		// 标识是否正常处理了
		return processed;
	}


	std::ofstream & GridSlamProcessor::outputStream() {
		return m_outputStream;
	}

	std::ostream & GridSlamProcessor::infoStream() {
		return m_infoStream;
	}


	int GridSlamProcessor::getBestParticleIndex()

	const {
		unsigned int	bi	= 0;
		double			bw	= -std::numeric_limits < double >::max();

		for (unsigned int i = 0; i < m_particles.size(); i++)
			if (bw < m_particles[i].weightSum) {
				bw					= m_particles[i].weightSum;
				bi					= i;
			}

		return (int)
		bi;
	}

	void GridSlamProcessor::onScanmatchUpdate() {
	}
	void GridSlamProcessor::onResampleUpdate() {
	}
	void GridSlamProcessor::onOdometryUpdate() {
	}


};


// end namespace
