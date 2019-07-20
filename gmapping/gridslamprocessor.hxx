

#ifdef MACOSX

// This is to overcome a possible bug in Apple's GCC.
#define isnan(x)				(x==FP_NAN)
#endif

/**Just scan match every single particle.
If the scan matching fails, the particle gets a default likelihood.*/
// 遍历粒子集，通过对初始位姿添加扰动，然后根据得分得到最优的位姿，然后更新粒子权重
// 然后计算在最优位姿下的活跃栅格点
inline void GridSlamProcessor::scanMatch(const double * plainReading)
{
	// sample a new pose from each scan in the reference
	double			sumScore = 0;

	// 遍历粒子集
	for (ParticleVector::iterator it = m_particles.begin(); it != m_particles.end(); it++) {
		OrientedPoint	corrected;
		double			score, l, s;

		// 通过对初始位姿添加扰动得到最优的位姿，记录在 corrected 中
		score				= m_matcher.optimize(corrected, it->map, it->pose, plainReading);

		//	  it->pose=corrected;
		// 当得分符合最小阈值要求时，就真正更新该位姿到粒子中
		if (score > m_minimumScore) {
			it->pose			= corrected;
		}
		else {
			if (m_infoStream) {
				m_infoStream << "Scan Matching Failed, using odometry. Likelihood=" << l << std::endl;
				m_infoStream << "lp:" << m_lastPartPose.x << " " << m_lastPartPose.y << " " << m_lastPartPose.theta << std::endl;
				m_infoStream << "op:" << m_odoPose.x << " " << m_odoPose.y << " " << m_odoPose.theta << std::endl;
			}
		}

		// s 记录得分，l 记录似然的指数部分
		m_matcher.likelihoodAndScore(s, l, it->map, it->pose, plainReading);

		// 累计所有粒子的得分
		sumScore			+= score;

		// 更新权重
		it->weight			+= l;

		// 累计粒子权重总和
		it->weightSum		+= l;

		//set up the selective copy of the active area
		//by detaching the areas that will be updated
		// 设置需要更新有效区域，即有效标识为 false
		m_matcher.invalidateActiveArea();

		// 根据地图、更新后的位姿、测距数据更新占用栅格
		m_matcher.computeActiveArea(it->map, it->pose, plainReading);
	}

	if (m_infoStream)
		m_infoStream << "Average Scan Matching Score=" << sumScore / m_particles.size() << std::endl;
}


inline void GridSlamProcessor::normalize()
{
	//normalize the log m_weights
	double			gain = 1./ (m_obsSigmaGain * m_particles.size());
	double			lmax = -std::numeric_limits <double>::max();

	// 找出最大权重
	for (ParticleVector::iterator it = m_particles.begin(); it != m_particles.end(); it++) {
		lmax				= it->weight > lmax ? it->weight: lmax;
	}

	//cout << "!!!!!!!!!!! maxwaight= "<< lmax << endl;
	m_weights.clear();
	double			wcum = 0;

	m_neff				= 0;

	// 计算粒子权重，并计算出所有粒子权重的和
	/*权重以最大权重为中心的高斯分布*/
	for (std::vector <Particle>::iterator it = m_particles.begin(); it != m_particles.end(); it++) {
		m_weights.push_back(exp(gain * (it->weight - lmax)));
		wcum				+= m_weights.back();

		//cout << "l=" << it->weight<< endl;
	}

	m_neff				= 0;

	// 计算粒子有效指标，该变量用于判断是否需要重采样
	for (std::vector <double>::iterator it = m_weights.begin(); it != m_weights.end(); it++) {
		// 归一化权重
		*it 				= *it / wcum;
		double			w	= *it;

		m_neff				+= w * w;
	}

	m_neff				= 1./ m_neff;

}


inline bool GridSlamProcessor::resample(const double * plainReading, int adaptSize, const RangeReading *)
{

	bool			hasResampled = false;

	// 定义：typedef std::vector<GridSlamProcessor::TNode*> TNodeVector
	TNodeVector 	oldGeneration;

	// 把每个粒子的最新节点复制出来
	for (unsigned int i = 0; i < m_particles.size(); i++) {
		oldGeneration.push_back(m_particles[i].node);
	}

	// 在 normalize 中计算的 m_neff
	if (m_neff < m_resampleThreshold * m_particles.size()) {

		if (m_infoStream)
			m_infoStream << "*************RESAMPLE***************" << std::endl;

		uniform_resampler <double, double> resampler;

		// 返回重采样后的粒子索引集合
		m_indexes			= resampler.resampleIndexes(m_weights, adaptSize);

		if (m_outputStream.is_open()) {
			m_outputStream << "RESAMPLE " << m_indexes.size() << " ";

			for (std::vector <unsigned int>::const_iterator it = m_indexes.begin(); it != m_indexes.end(); it++) {
				m_outputStream << *it << " ";
			}

			m_outputStream << std::endl;
		}

		// 空函数
		onResampleUpdate();

		//BEGIN: BUILDING TREE
		ParticleVector	temp;
		unsigned int	j	= 0;

		std::vector <unsigned int> deletedParticles; //this is for deleteing the particles which have been resampled away.

		//		cerr << "Existing Nodes:" ;
		for (unsigned int i = 0; i < m_indexes.size(); i++) {
			//			cerr << " " << m_indexes[i];
			// 需要删除的粒子的 index
			while (j < m_indexes[i]) {
				deletedParticles.push_back(j);
				j++;
			}

			if (j == m_indexes[i])
				j++;
			// 需要保留的 粒子
			Particle &		p	= m_particles[m_indexes[i]];
			TNode * 		node = 0;
			// 上一时刻的粒子
			TNode * 		oldNode = oldGeneration[m_indexes[i]];

			//			cerr << i << "->" << m_indexes[i] << "B("<<oldNode->childs <<") ";
			// 申请新的粒子空间
			node				= new TNode(p.pose, 0, oldNode, 0);
			// 由于是为了下一时刻用的，所以还没有可用的 scan 数据
			node->reading		= 0;

			//			cerr << "A("<<node->parent->childs <<") " <<endl;
			// 采样后需要保留下来的粒子
			temp.push_back(p);
			// 指向新申请的粒子节点空间
			temp.back().node	= node;
			temp.back().previousIndex = m_indexes[i];
		}
		// 
		while (j < m_indexes.size()) {
			deletedParticles.push_back(j);
			j++;
		}

		//		cerr << endl;
		std::cerr << "Deleting Nodes:";
		// 删除 没有采样到的粒子
		for (unsigned int i = 0; i < deletedParticles.size(); i++) {
			std::cerr << " " << deletedParticles[i];
			// 删除该粒子绑定的节点
			delete m_particles[deletedParticles[i]].node;
			m_particles[deletedParticles[i]].node = 0;
		}

		std::cerr << " Done" << std::endl;

		//END: BUILDING TREE
		std::cerr << "Deleting old particles...";
		// 清空 m_particles 中记录的粒子集，新的粒子集都在 temp 中
		m_particles.clear();
		std::cerr << "Done" << std::endl;
		std::cerr << "Copying Particles and  Registering  scans...";
		// 从 temp 中读取新采样的粒子集
		for (ParticleVector::iterator it = temp.begin(); it != temp.end(); it++) {
			// 权重 置 0
			it->setWeight(0);
			m_matcher.invalidateActiveArea();
			// 更新该粒子
			m_matcher.registerScan(it->map, it->pose, plainReading);
			// 新的粒子集复制到 m_particles
			m_particles.push_back(*it);
		}

		std::cerr << " Done" << std::endl;
		hasResampled		= true;
	}
	// 不重采样时执行
	else {
		int 			index = 0;

		std::cerr << "Registering Scans:";
		TNodeVector::iterator node_it = oldGeneration.begin();
		// 给每个粒子增加新的节点 node
		for (ParticleVector::iterator it = m_particles.begin(); it != m_particles.end(); it++) {
			//create a new node in the particle tree and add it to the old tree
			//BEGIN: BUILDING TREE	
			TNode * 		node = 0;

			node				= new TNode(it->pose, 0.0, *node_it, 0);

			node->reading		= 0;
			it->node			= node;

			//END: BUILDING TREE
			m_matcher.invalidateActiveArea();
			m_matcher.registerScan(it->map, it->pose, plainReading);
			it->previousIndex	= index;
			index++;
			node_it++;

		}

		std::cerr << "Done" << std::endl;

	}

	//END: BUILDING TREE
	return hasResampled;
}


