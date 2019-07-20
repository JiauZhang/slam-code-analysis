#ifndef HARRAY2D_H
#define HARRAY2D_H
#include <set>
#include <utils/point.h>
#include <utils/autoptr.h>
#include "array2d.h"

namespace GMapping {

template <class Cell>
class HierarchicalArray2D: public Array2D<autoptr< Array2D<Cell> > >{
	public:
		typedef std::set< point<int>, pointcomparator<int> > PointSet;
		HierarchicalArray2D(int xsize, int ysize, int patchMagnitude=5);
		HierarchicalArray2D(const HierarchicalArray2D& hg);
		HierarchicalArray2D& operator=(const HierarchicalArray2D& hg);
		virtual ~HierarchicalArray2D(){}
		void resize(int ixmin, int iymin, int ixmax, int iymax);
		inline int getPatchSize() const {return m_patchMagnitude;}
		inline int getPatchMagnitude() const {return m_patchMagnitude;}
		
		inline const Cell& cell(int x, int y) const;
		inline Cell& cell(int x, int y);
		inline bool isAllocated(int x, int y) const;
		inline AccessibilityState cellState(int x, int y) const ;
		inline IntPoint patchIndexes(int x, int y) const;
		
		inline const Cell& cell(const IntPoint& p) const { return cell(p.x,p.y); }
		inline Cell& cell(const IntPoint& p) { return cell(p.x,p.y); }
		inline bool isAllocated(const IntPoint& p) const { return isAllocated(p.x,p.y);}
		inline AccessibilityState cellState(const IntPoint& p) const { return cellState(p.x,p.y); }
		inline IntPoint patchIndexes(const IntPoint& p) const { return patchIndexes(p.x,p.y);}
		
		inline void setActiveArea(const PointSet&, bool patchCoords=false);
		const PointSet& getActiveArea() const {return m_activeArea; }
		inline void allocActiveArea();
	protected:
		virtual Array2D<Cell> * createPatch(const IntPoint& p) const;
		PointSet m_activeArea;
		int m_patchMagnitude;
		int m_patchSize;
};

template <class Cell>
HierarchicalArray2D<Cell>::HierarchicalArray2D(int xsize, int ysize, int patchMagnitude) 
  :Array2D<autoptr< Array2D<Cell> > >::Array2D((xsize>>patchMagnitude), (ysize>>patchMagnitude)){
	m_patchMagnitude=patchMagnitude;
	m_patchSize=1<<m_patchMagnitude;
}

template <class Cell>
HierarchicalArray2D<Cell>::HierarchicalArray2D(const HierarchicalArray2D& hg)
  :Array2D<autoptr< Array2D<Cell> > >::Array2D((hg.m_xsize>>hg.m_patchMagnitude), (hg.m_ysize>>hg.m_patchMagnitude))  // added by cyrill: if you have a resize error, check this again
{
	this->m_xsize=hg.m_xsize;
	this->m_ysize=hg.m_ysize;
	this->m_cells=new autoptr< Array2D<Cell> >*[this->m_xsize];
	for (int x=0; x<this->m_xsize; x++){
		this->m_cells[x]=new autoptr< Array2D<Cell> >[this->m_ysize];
		for (int y=0; y<this->m_ysize; y++)
			this->m_cells[x][y]=hg.m_cells[x][y];
	}
	this->m_patchMagnitude=hg.m_patchMagnitude;
	this->m_patchSize=hg.m_patchSize;
}

template <class Cell>
void HierarchicalArray2D<Cell>::resize(int xmin, int ymin, int xmax, int ymax){
	int xsize=xmax-xmin;
	int ysize=ymax-ymin;
	autoptr< Array2D<Cell> > ** newcells=new autoptr< Array2D<Cell> > *[xsize];
	for (int x=0; x<xsize; x++){
		newcells[x]=new autoptr< Array2D<Cell> >[ysize];
		for (int y=0; y<ysize; y++){
			newcells[x][y]=autoptr< Array2D<Cell> >(0);
		}
	}
	int dx= xmin < 0 ? 0 : xmin;
	int dy= ymin < 0 ? 0 : ymin;
	int Dx=xmax<this->m_xsize?xmax:this->m_xsize;
	int Dy=ymax<this->m_ysize?ymax:this->m_ysize;
	for (int x=dx; x<Dx; x++){
		for (int y=dy; y<Dy; y++){
			newcells[x-xmin][y-ymin]=this->m_cells[x][y];
		}
		delete [] this->m_cells[x];
	}
	delete [] this->m_cells;
	this->m_cells=newcells;
	this->m_xsize=xsize;
	this->m_ysize=ysize; 
}

template <class Cell>
HierarchicalArray2D<Cell>& HierarchicalArray2D<Cell>::operator=(const HierarchicalArray2D& hg){
//	Array2D<autoptr< Array2D<Cell> > >::operator=(hg);
	if (this->m_xsize!=hg.m_xsize || this->m_ysize!=hg.m_ysize){
		for (int i=0; i<this->m_xsize; i++)
			delete [] this->m_cells[i];
		delete [] this->m_cells;
		this->m_xsize=hg.m_xsize;
		this->m_ysize=hg.m_ysize;
		this->m_cells=new autoptr< Array2D<Cell> >*[this->m_xsize];
		for (int i=0; i<this->m_xsize; i++)
			this->m_cells[i]=new autoptr< Array2D<Cell> > [this->m_ysize];
	}
	for (int x=0; x<this->m_xsize; x++)
		for (int y=0; y<this->m_ysize; y++)
			this->m_cells[x][y]=hg.m_cells[x][y];
	
	m_activeArea.clear();
	m_patchMagnitude=hg.m_patchMagnitude;
	m_patchSize=hg.m_patchSize;
	return *this;
}


template <class Cell>
void HierarchicalArray2D<Cell>::setActiveArea(const typename HierarchicalArray2D<Cell>::PointSet& aa, bool patchCoords){
	m_activeArea.clear();
	// m_activeArea 记录新的处于活跃状态的 patch index
	for (PointSet::const_iterator it= aa.begin(); it!=aa.end(); it++){
		IntPoint p;
		if (patchCoords)
			p=*it;
		else
			p=patchIndexes(*it);
		m_activeArea.insert(p);
	}
}

template <class Cell>
// 分配 32*32 的 cell 栅格
Array2D<Cell>* HierarchicalArray2D<Cell>::createPatch(const IntPoint& ) const{
	return new Array2D<Cell>(1<<m_patchMagnitude, 1<<m_patchMagnitude);
}


template <class Cell>
AccessibilityState  HierarchicalArray2D<Cell>::cellState(int x, int y) const {
	if (this->isInside(patchIndexes(x,y)))
		if(isAllocated(x,y))
			return (AccessibilityState)((int)Inside|(int)Allocated);
		else
			return Inside;
	return Outside;
}

template <class Cell>
void HierarchicalArray2D<Cell>::allocActiveArea(){
	// 遍历激光雷达扫描覆盖的栅格点坐标
	for (PointSet::const_iterator it= m_activeArea.begin(); it!=m_activeArea.end(); it++){
		// 取出地图栅格中该点存放的 cell
		// 判断 m_activeAera 中记录的处于活跃状态的 patch index 是否已分配内存
		const autoptr< Array2D<Cell> >& ptr=this->m_cells[it->x][it->y];
		Array2D<Cell>* patch=0;
		// 如果该坐标栅格下还没有分配 cell 数据结构空间，则申请，并绑定该坐标
		// 指定 patch 未分配内存时执行
		if (!ptr){
			patch=createPatch(*it);
		} else{	
			// 如果已有，则开辟新的空间复制过来
			// 已存在时，开辟新的空间，并将原来的复制过来
			patch=new Array2D<Cell>(*ptr);
		}
		// 绑定新的
		// 更新新开辟的 patch 空间，由于 autoptr 的存在，所以原来的 patch 会自动释放
		this->m_cells[it->x][it->y]=autoptr< Array2D<Cell> >(patch);
	}
}

template <class Cell>
bool HierarchicalArray2D<Cell>::isAllocated(int x, int y) const{
	IntPoint c=patchIndexes(x,y);
	autoptr< Array2D<Cell> >& ptr=this->m_cells[c.x][c.y];
	return (ptr != 0);
}

template <class Cell>
// 返回指定栅格所在 patch index
IntPoint HierarchicalArray2D<Cell>::patchIndexes(int x, int y) const{
	if (x>=0 && y>=0)
		return IntPoint(x>>m_patchMagnitude, y>>m_patchMagnitude);
	return IntPoint(-1, -1);
}

template <class Cell>
Cell& HierarchicalArray2D<Cell>::cell(int x, int y){
	// 地图栅格的管理是以 32*32个栅格 组成一个 patch 管理的
	IntPoint c=patchIndexes(x,y);
	assert(this->isInside(c.x, c.y));
	if (!this->m_cells[c.x][c.y]){
		Array2D<Cell>* patch=createPatch(IntPoint(x,y));
		this->m_cells[c.x][c.y]=autoptr< Array2D<Cell> >(patch);
		//cerr << "!!! FATAL: your dick is going to fall down" << endl;
	}
	// 指定 cell 所在 patch，即小的 二维栅格数组
	autoptr< Array2D<Cell> >& ptr=this->m_cells[c.x][c.y];
	// 返回该 cell 的引用
	return (*ptr).cell(IntPoint(x-(c.x<<m_patchMagnitude),y-(c.y<<m_patchMagnitude)));
}

template <class Cell>
const Cell& HierarchicalArray2D<Cell>::cell(int x, int y) const{
	assert(isAllocated(x,y));
	IntPoint c=patchIndexes(x,y);
	const autoptr< Array2D<Cell> >& ptr=this->m_cells[c.x][c.y];
	return (*ptr).cell(IntPoint(x-(c.x<<m_patchMagnitude),y-(c.y<<m_patchMagnitude)));
}

};

#endif

