# 第一章 绪论

## 基本概念和术语

**数据(Data)**：是对信息的一种符号表示。在计算机科学中是指所有能输入到计算机中并被计算机程序处理的符号的总称。

**数据元素(Data Element)**：是数据的基本单位，在计算机程序中通常作为一个整体进行考虑和处理。

**数据项**(Data Item)：一个数据元素可由若干个数据项组成。数据项是数据的不可分割的最小单位。

**数据对象(**Data Object)：是性质相同的数据元素的集合。是数据的一个子集。例：整数数据对象
是集合{0,±1, ±2, ±3,......}

**关键字**：指的是能识别一个或多个数据元素的数据项。若能起唯一识别作用，则称之为 "主" 关
键字，否则称之为 "次" 关键字。

数据元素之间的4类基本结构：集合结构、线性结构、树状结构、图状或网状结构。

**数据结构的形式定义为：**

- 数据结构是一个二元组：Data_Structures=(D,S)
- 其中：D是数据元素的有限集，S是D上关系的有限集。
- 上述数据结构定义仅是对操作对象的数学描述。结构定义中的“关系”描述的是数据元素之间的逻辑关系，即“逻辑结构”。
- 数据结构在计算机中的表示称为数据的物理结构，又称为存储结构。存储结构是逻辑结构在存储器中的映象，包括数据元素的表示和关系的表示。

数据元素之间的关系有两种表示方法：顺序映像和非顺序映像，由此得到两种存储结构：**顺序存储结构和链式存储结构**。

- 顺序存储映像的特点：借助元素在存储器之间的**相对位置**来表示数据元素之间的逻辑关系。
- 非顺序存储映像的特点：借助指示元素存储位置的**指针**来表示元素之间的逻辑关系。

## 抽象数据类型的表示与实现

**抽象数据类型**（Abstract Data Type 简称 ADT）是指一个数学模型以及定义在此数学模型上的一组操作。

**数据抽象：**抽象数据类型的定义仅取决于它的逻辑特性，`与其在计算机内部如何表示和实现无关`。即，与硬件平台、存储结构、算法和编程语言等等无关。

**数据封装：**将实体的外部特性和其内部实现细节分离，并且对外部用户隐藏其内部实现细节。

抽象数据类型的形式描述为：

- ADT = ( D，S，P )
- 其中：D 是数据对象，S 是 D 上的关系集，P 是 D 的基本操作集。

## 算法和算法分析



# 第二章 线性表



## 2.1 线性表的类型定义

**线性表(Linear List) ：**由n(n≧0)个数据元素(结点)a1，a2，...an组成的有限序列。

- 其中数据元素的个数n定义为表的长度。 （有n-1个有序对）
- 当n=0时，称为空表，常常将非空的线性表(n>0)记作：
  (a1，a2，...an)

**逻辑特征：**

- 有且仅有一个开始结点a1，它没有直接前趋，而仅有一个直接后继a2；
- 有且仅有一个终端结点an，它没有直接后继，而仅有一个直接前趋an-1；
- 其余的内部结点ai (2≦i≦n-1) 都有且仅有一个直接前趋 ai-1和一个直接后继 ai+1。

**引用型操作：**“L”是“传值参数”。“引用” 传递参数“ L ”，即在参数 L 之前加有符号“ & ” 。

- ListEmpty(L)
- ListLength(L)
- PriorElem(L,cur_e,&pre_e):若cur_e是L中的数据元素，则用pre_e返回它的前驱，否则操作失败，pre_e 无定义。
- NextElem(L,cur_e,&next_e)：若cur_e是L中的数据元素，则用next_e返回它的后继，否则操作失败，next_e 无定义。
- GetElem(L,i,&e)：用 e 返回 L 中第 i 个元素的值。
- LocateElem(L,e,compare())：返回 L 中第1个与 e 满足关系 compare( ) 的元素的位序。若这样的元素不存在，则返回值为0。
- ListTraverse(L,visit())

**加工型操作：**

- ClearList(&L)
- PutElem(&L,i,&e)：L 中第 i 个元素赋值同 e 的值。
- ListInsert(&L,i,e)：在 L 的第 i 个元素之前插入新的元素 e，L 的长度增1。
- ListDelete(&L,i,&e)：删除 L 的第 i 个元素，并用 e 返回其值，L 的长度减1。

### 相关算法实现

已知集合 A 和 B，求两个集合的并集，使 A＝A∪B。

已知线性表LA和线性表LB中的数据元素按值非递减有序排列，现要求将LA和LB归并为一个新的线性表LC，且LC中的元素仍按值非递减有序排列。

已知一个“非纯集合”B，试构造一个集合 A，使 A 中只包含 B 中所有值各不相同的数据元素。

判别两个集合是否相等。

## 2.2 线性表的顺序表示和实现

### 顺序表定义

**顺序表**是线性表的顺序存储表示的简称

- "用一组地址连续的存储单元依次存放线性表中的数据元素"，即以**"存储位置相邻"**表示"位序相继的两个数据元素之间的**前驱和后继的关系**(有序对< ，>)"
- 以表中第一个元素的存储位置作为线性表的起始地址，称作线性表的基地址。

![image-20241215150910220](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241215150910220.png)

### 数据元素存储

假设每个数据元素占据的存储量是一个常量 C，
则 后 继 元 素 的 存 储 地 址 和 其 前 驱 元 素 相 隔 一 个 常 量 ，

即：**LOC( a i ) = LOC( a i-1 ) + C**

所有数据元素的存储位置均可由第一个数据元素的存储位置得到: 

**LOC( a i ) = LOC( a 1 ) + ( i -1 )×C**

### 顺序表数据成员

存储元素的一维数组空间、指示顺序表中已有的元素个数、存放的数据元素个数的最大值

![image-20241215151344526](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241215151344526.png)

```
// 存储结构
const int LIST_INIT_SIZE=100； // 线性表存储空间初始分配量
const int LISTINCREMENT=10； // 线性表存储空间分配增量
typedef struct {
ElemType *elem； // 存储空间基址
int length； // 当前长度
int listsize； // 允许的最大存储容量
// (以sizeof(ElemType)为单位)
} SqList； // 顺序表
```

### 顺序表中基本操作的实现

一、初始化操作

二、元素定位操作

三、插入元素操作

四、删除元素操作

五、销毁结构操作

## 2.3 线性表的链式表示与实现

### **线性表的链式存储表示的特点与定义**

- 是用一组任意的（可以是连续的，也可以是不连续的）存储单元存储线性表 的数据元素。
- 对数据元素 ai 来说，除了存储其本身的信息之外，还需存储一个指示其直接后继的信息(即直接后继的存储位置)。两部分信息组成一个“结点”。
- 由分别表示 a1 , a2 ,…, an 的 N 个结点依次相链构成的链表， 称为线性表的链式存储表示。此类链表的每个结点中只包含一个指针域，故又称单 链表或线性链表。

![image-20241215151750697](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241215151750697.png)

![image-20241215151832180](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241215151832180.png)

### 单链表和指针

**头指针**：在链式存储结构中仍以第一个数据元素的 存储地址作为线性表的基地址，通常称它为头指针，线性表中所有数据元素都可以从头指针出发找到。

**空指针**：因为线性表的最后一个数据元素没有后继，因此，在单链 表中最后一个结点中的“指针”是一个特殊的值 “NULL” 。

为了便于处理一些特殊情况，在第一个结点之前附加一个“头结点” （不具有数据），令该结点中指针域的指针指向第一个元素结点。

若线性表为空，在不带头结点的情况下，头指针为空 (NULL)。

```
typedef struct_LNode{
	ElemType data;
	struct_LNode *next;
}LNode,*LinkList;
```

链表是一个进行动态存储管理的结构，因此在初始化时不需要按照线性表实际所需最大容量进行预分配。

### 单链表中基本操作的实现

一、初始化操作

```
void initList(LinkList &L){
	L = new LNode;
	if(!L) exit(1);
	L->next = NULL;
}
```

二、销毁结构操作

从头结点开始依次访问链表各节点，释放存储空间

```
// 销毁以L为头指针的单链表，释放链表中所有结点空间,包括头结点
void DestroyList(LinkList &L){
	LinkList p = L;
	while(p){
		pTemp = p;
        p = p->next;
        delete pTemp;
	}
	p = NULL;
}

// 将单链表清为空表，仅保留头结点，释放链表中其他结点空间
void DestroyList(LinkList &L){
	LinkList p = L->next;
	while(p){
		pTemp = p;
        p = p->next;
        delete pTemp;
	}
}
```

三、存取元素操作

单链表是一种“顺序存取”的结构，即：为取第 i 元素，首先必须先找到第 i-1 个元 素。因此不论 i 值为多少，都必须从头结点开始，依次访问，直数到第 i 个为止。头 结点可看成是第位序为0的结点。

```
Status GetElem(LinkList L,ElemType &e, int pos){
	j = 1;
	p = L->next;
	while(j<pos && p){
		p = p->next;
		++j;
	}
	if(!p || j>pos) return ERROR;
	e = p->data;
	return OK;
}
```

四、插入元素操作

首先找到第i-1个结点，然后修改相应 指针。

```
Status ListInsert(LinkList &L,ElemType e, int pos){
	p=L,j=0;
    while (p && j<pos-1){
    	p = p->next;
    	++j;
    }
    if(!p || j>pos-1) return ERROR;
    s = new LNode;
    if(!s) exit(OVERFLOEW);
    s->data=e;
    s->next=p->next;
    p->next=s;
    return OK;
}
```

五、删除元素操作

```
Status ListDelete(LinkList &L,ElemType e, int pos){
	p=L,j=0;
    while (p && j<pos-1){
    	p = p->next;
    	++j;
    }
    if(!p || j>pos-1) return ERROR;
    q=p->next;
    p->next=q->next;
    e=q->data;
    delete q;
    return OK;
}
```

### 单链表其它算法举例

**逆序创建链表**

假设线性表( a1 , a2 ,…, aN )的数据元素存储在一维数组 A[n] 中，则从数组的最后一个分量起，依次生成结点，并逐个插入 到一个初始为“空”的链表中。

```
void CreateList_L(LinkList &L， int n， ElemType A[ ])
{ 
    L = new LNode；
    if (!L) exit(OVERFLOW)；
    L->next = NULL；点的空的单链表
    for (i = n； i > 0； --i){ 
    	p = new LNode；
        if (!p) exit(OVERFLOW)； 
        p->data = A[i-1]；
        p->next = L->next；
        L->next = p； 
    }
} // CreateList_L
```



 以链表作存储结构解例2-5的问题，即将线性表 ( a1 , a2 ,…, am, b1 , b2 ,…, bn ) 改变成( b1 , b2 ,…, bn , a1 , a2 ,…, am ) 。

```
void exchange_L(LinkList &L,int m){
	// 本算法实现单链表中前 m 个结点和后 n 个结点的互换
	if ( m && L->next ){
            p=L->next,k=1;
        while (k<m && m){
            p=p->next;
            ++k;
        }
        if (p && p->next){
			ha=L->next;
			L->next = p->next;
			p->next=NULL;
			q=L->next
			while(q->next){
				q=q->next;
			}
			q->next=ha;
        }
	}
}
```



编写算法删除单链表中"多余"的数据元素，即使操作之后的单链表中所有元素的值都不相同。

```
⚫void purge_L(LinkList &L )
```

### 静态链表

用数组描述的链表 

- 在不设指针类型的高级程序设计语言中使用链表结构。

- 数组的第零分量表示头结点，每个结点的指针域表示下一 结点在数组中的相对位置。

- ```
  #define MAXSIZE 1000
  typedef struct {
  ElemType data；
  int cur；
  } component，SLinkList[MAXSIZE];
  ```

### 循环链表

循环链表(Circular Linked List)是线性表的另一种形式的链式存 表示。

它的特点是表中最后一个结点的指针域指向头结点，整 个链表成为一个由链指针相链接的环，并且设立尾指针设成指向最后一个结点。

空的循环链表由只含一个自成循环的头结点表示。

循环链表的操作和单链表基本一致，差别仅在于，**判别链表中最 后一个结点的条件不再是"后继是否为空"，而是"后继是否为头结点 "。**

### 双向链表

双向链表(Double Linked List)的特点是其结点结构中含有两 个指针域，其一指向数据元素的“直接后继”，另一指向数据 元素的“直接前驱”。

```
typedef struct _DuLNode{
    ElemType data；
    struct _DuLNode *prior；
    struct _DuLNode *next；
} DuLNode，*DuLink；
```

## 2.4 一元多项式的表示及相加

![image-20241215160138812](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241215160138812.png)

![image-20241215160231564](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241215160231564.png)

## 2.5 有序表

### 有序表的定义

若线性表中的数据元素相互之间可以比较，并且数据元素在线 性表中依值非递减或非递增有序排列，即ai≥ai-1或ai≤ai-1 (i = 2,3,…,n)，则称该线性表为有序表(Ordered List)。

#### 例题

已知两个链表（头指针分别为 La 和 Lb）中的数据元素 均自小至大有序，编写算法将这两个链表归并为一个链表。

做法一：将 Lb 表中的各个结点插入 到 La 表中的相应位置中去。即按照有序关系首先查找插入位 置，然后修改相应指针。

做法二：试设想新建一个空的链表，然后将已知两个链表中的结点依从 小到大的次序逐个插入到这个新的链表中。

```
void MergeList(LinkList &La,LinkList &Lb,LinkList &Lc){
	pa=La->next;
	pb=Lb->next;
	Lc=rc=new LNode;
	while(pa && pb){
		if(pa->data<=pb->data){
			rc->next=pa;
			rc=pa;
			pa=pa->next;
		}else{
			rc->next=pb;
			rc=pb;
			pb=pb->next;
		}
	}
	if(pa){pc->next=pa;}
	if(pb){pc->next=pb;}
}
```

### 有序表的基本操作

有序表类型的基本操作和线性表类型中定义的基本操作基本相同， 但由于 LocateElem 中函数参数 compare 的类型不同:

LocateElem( L, e, &q, int(*compare)(ElemType,ElemType) )

若有序表L中存在元素 e，则 q 指示L中第一个值为 e 的元素的位 置，并返回函数值TRUE；否则 q 指示第一个大于 e 的元素的前驱的位置，并 返回函数值 FALSE。

### 有序链表类型

```
// 结构定义
typedef struct LNode { // 结点结构
    ElemType data;
    struct LNode *next;
} *SLink;

typedef struct { // 链表结构
    SLink head, // 指向有序链表中的头结点
    tail, // 指向有序链表中最后一个结点
    curPtr; // 指向操作的当前结点，称为"当前指针"
    int length, // 指示有序链表的长度
    curPos; // 指示当前指针所指结点的位序
} 
```

### 基本操作接口

**bool MakeNode( SLink &p, ElemType e );** 

生成一个数据元素和 e 相同的新结点 *p

**bool InitList(OrderedLinkList &L );**

构造一个空的有序链表L

**void DestroyList(OrderedLinkList &L );** 

销毁有序链表 L，L 不再存在。

**bool ListEmpty (OrderedLinkList L );** 

若有序链表 L 为空表，则返回TRUE，否则返回 FALSE。 

**int ListLength(OrderedLinkList L );**

返回有序链表L中元素个数。

**SLink PriorPos(OrderedLinkList L );** 

移动有序链表L中当前指针到它当前所指结点的直接前驱并返回。

**SLink NextPos (OrderedLinkList L );**

移动有序链表L中当前指针到它当前所指结点的直接后继并返回。

**bool GetPos (OrderedLinkList L, int pos ) ;** 

若1≤pos≤LengthList(L)，则移动当前指针指向第 pos 个结点。

**void GetCurElem(OrderedLinkList L, ElemType& e );** 

**以 e 带回当前指针所指结点中的数据元素。**

**bool LocateElem (OrderedLinkList L, ElemType e,int (*compare)(ElemType, ElemType));**

若有序链表L中存在和e相同的数据元素,则当前指针指向第1个和e相同的 结点，并返回TRUE，否则当前指针指向第一个大于e的元素的前驱，并返回 FALSE。

**void ListTraverse(OrderedLinkList L, void (*visit)() );** 

依次对L的每个元素调用函数 visit()。

**void ClearList(OrderedLinkList &L );**

将有序链表L重置为空表，并释放原链表的结点空间。

**void SetcurElem(OrderedLinkList L , ElemType e );**

将有序链表L中当前指针所指结点中的数据元素修改为和 e 相同。

**void InsAfter (OrderedLinkList &L, SLink s );** 

在有序链表L中当前指针所指结点之后插入一个新的结点 *s ,并移动当前指针指向新插入的结点。

**bool DelAfter(OrderedLinkList &L, ElemType& e );** 

若当前指针所指非单链表L中最后一个结点，则删除当前指针所指结点之 后的结点,以e带回它的数据元素并返回TRUE,否则不进行删除操作且返回 FALSE。

### 有序表的应用例子

假设以两个有序表分 别表示集合A和B，试求集 合C=A∪B。

![image-20241215164907637](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241215164907637.png)

假设以两个有序表分别表示集合A和B，试求集合 C=A∩B。

```
void Intersection (OrderLinkList A, OrderLinkList B, OrderLinkList &C)
{
if ( InitList(C) ) {
    m = ListLength(A); n = Listlength(B); 
    i = 1; j = 1;
    while ( i <= m && j <= n ) {
        if ( GetPos(A,i) && GetPos(B,j) ) {
        	GetCurElem( A,ea ); 
        	GetCurElem( B,eb );
           	if ( ea < eb ) ++i;
            else if ( ea > eb ) ++j;
            else {
            	if ( !MakeNode( s,ea ) ) exit(1);
            	++i;++j;
            	InsAfter(C,s);
    } 
} // Intersection
```



# 第三章 栈和队列

## 3.1 栈 

### 栈的类型定义

**栈(Stack)**是限定只能在表的一端进行插入和删除操作的线性表。

在表中，允许插入和删除的一端称作**“栈顶(top)”**，不允许插入和删除的另 一端称作**“栈底(bottom)”** 。

当表中没有元素时称为**空栈**。

栈称为后进先出表（LIFO）。

```
ADT Stack {
基本操作：

InitStack(&S)
操作结果：构造一个空栈 S。

DestroyStack(&S)
初始条件：栈 S 已存在。
操作结果：栈 S 被销毁。

ClearStack(&S)
初始条件：栈 S 已存在。
操作结果：将 S 清为空栈。

StackEmpty(S)
初始条件：栈 S 已存在。
操作结果：若栈 S 为空栈，则返回TRUE，否则返回FALSE。

StackLength(S)
初始条件：栈 S 已存在。
操作结果：返回栈 S 中元素个数，即栈的长度。

GetTop(S, &e)
初始条件：栈 S 已存在且非空。
操作结果：用 e 返回S的栈顶元素。

Push(&S, e)
初始条件：栈 S 已存在。
操作结果：插入元素 e 为新的栈顶元素。

Pop(&S, &e)
初始条件：栈 S 已存在且非空。
操作结果：删除 S 的栈顶元素，并用 e 返回其值。

StackTraverse(S, visit( ))
初始条件：栈 S 已存在且非空，visit( )为元素的访问函数。
操作结果：从栈底到栈顶依次对S的每个元素调用函数visit( )，一旦
visit( )失败，则操作失败。

} ADT Stack

```

### 栈的存储表示和操作的实现

#### **顺序栈**

栈也有顺序存储和链式存储两种存储表示，其顺序存储结构简称为**顺序栈**。

```
#define STACK_INIT_SIZE 100 //栈空间初始容量，以元素为单位
#define STACK_INCREMENT //栈空间扩容增量，以元素为单位
typedef struct {
SElemType *base; // 栈底指针：存储空间基址
SElemType *top; // 栈顶指针
int stacksize; // 允许最大容量，以元素为单位
} SqStack;
```

![image-20241215170812473](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241215170812473.png)

```
Status InitStack(SqStack *S){
	S.base=(SElemType*)malloc(STACK_INIT_SIZE*sizeof(SElemType));
	if(S.base==NULL){
		exit(OVERFLOW);
	}
	S.stacksize=STACK_INIT_SIZE;
	S.top=S.base;
	return OK;
}

Status GetTop (SqStack S, ElemType &e){
	if(S.base==S.top){
		return ERROR;
	}
	e = *(S.top-1)
	return OK;
}

Status Push (SqStack &S, ElemType e){
	if(S.top - S.base >= S.stacksize) {
	S.base = (SElemType *)realloc(S.base,(S.stacksize + STACK_INCREMENT)*sizeof(SElemType));
	if(!S.base) exit(OVERFLOW);
	 //基地址可能变化，修正栈顶指针
	S.top=S.base+S.stacksize;
	//更新栈最大容量
	S.stacksize+=STACK_INCREMENT;
	}
	*(S.top++)=e;
	return OK;
}

Status Pop (SqStack &S, ElemType &e){
    if(S.base == S.top) return ERROR;
    e = *(--S.top);
    return OK;
}
```

#### 链栈

链栈即为栈的链式存储结构。

由于栈只在栈顶作插入和删除操作，因此链栈中不需要头结点。

链栈中指针的方向是从栈顶指向栈底 的，这正好和单链表是相反的。

![image-20241215171841484](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241215171841484.png)

```
typedef struct _Node{
    ElemType data;
    struct _Node *next;
    } Node, *Link;
    
typedef struct {
    Link top;
    int length;
} LnkStack;

Status InitStack (LnkStack &S){
	S.top=NULL;
	S.length=0;
}

Status Push ( Stack &S, ElemType e ){
	p = new Node;
	if(!p) exit(OVERFLOW);
	p->data=e;
	p->next=S.top;
	S.top=p;
	S.length++;
}

Status Pop ( LnkStack &S, SElemType &e ){
	if(!S.top)return ERROR;
    else{
    	e = S.top->data;
    	q = S.top;
    	S.top = S.top->next;
    	--S.length;
    	delete q;
    	return OK;
    }
}
```

## 3.2 栈的应用举例

##### 数制转换

```
void conversion(){
	initStack(S);
	scanf("%d",&N);
	while(N){
		Push(S,N%8);
		N = N/8;
	}
	while(!StackEmpty(S)){
		Pop(S,e);
		printf("%d",e);
	}
}
```

##### 括号匹配

##### 迷宫求解问题

求迷宫中一条路径的算法的基本思想是：

若当前位置“可通” ，则纳入“当前路径” ，并继续朝 “下一位置”探索；

若当前位置“不可通” ，则应顺着“来的方向”退回到 “前一通道块” ，然后朝着除“来向”之外的其他方向继续探索；

若该通道块的四周四个方块均“不可通” ，则应从“当前路径”上删除该通道块。

假设以栈S记录"当前路径" ，则栈顶中存放的是"当前路径 上最后一个通道块" 。 "纳入路径"的操作即为"当前位置入 栈" ；"从当前路径上删除前一通道块"的操作即为"出栈" 。

(等待)

```
bool MazePath(MazeType name, PosType start, PosType end){
//若在迷宫name中存在从start到end的路径，保存在栈中，并返回TRUE；
InitStack(S); curpos = start; curstep = 1;
do{
if(Pass(curpos)){//当前位置可行，不是砖、没走过、不是不可行块
FootPrint(curpos); //留下足迹，标记为“经过块”
e.seat = curpos; e.ord = curstep; e.di = 1;//{1,2,3,4}->{east,south,west,north}
Push(e);
if(seat == end){return TRUE;}
curpos = NextPos(curpos,1); curstep++; //得到下一个待检测的位置，从东方开始
}else{//当前位置不可行
if(!StackEmpty(S)){
Pop(S, e);
while(e.di == 4 && !StackEmpty(S)){
MarkPrint(e.seat); Pop(S, e);/*标记为不可行点*/}
if(e.di < 4){e.di++; Push(S,e); curpos = NextPos(e.seat, e.di);} //下一个待检测的位置
}//end of 栈不空
}//end of 当前位置不可行
}while(!StackEmpty(S)); return FALSE;
}
```

##### 表达式求值问题

称 OP + S1 + S2 为表达式的前缀表示法（简称前缀式） 

称 S1 + OP + S2 为表达式的中缀表示法（简称中缀式） 

称 S1 + S2 + OP 为表达式的后缀表示法（简称后缀式）

```
void transform(char suffix[],char exp[]){
// 从合法的表达式字符串 exp 求得其相应的后缀式suffix
	initStack(S);
	p=exp;
	ch=*p;
	while(!StackEmpty(S)){
		// 若当前字符是操作数，直接发送给后缀式
		if(!IN(ch,OP))Pass(suffix,ch);
		else{
            switch(ch){
				case '(':
                    Push(S,ch);
                    break;
				case ')':
					Pop(S,c);
					break;
				default:
				////持续传递栈顶元素，直到优先级低于当前操作符
					while(Gettop(S,c) && precede(c,ch)){
                        Pass(suffix,c);
                        Pop(S,ch);
                        if(ch!='#'){
							Push(S,ch);
							break;
                        }
					}
            }
		}
	}
	if(ch!='#'){
		p++;
		ch=*p;
	}
}
```

##### 递归函数的实现

递归过程执行过程中所占用的数据区，称之为递归工作栈。

每一层的递归参数等数据合成一个记录，称之为递归工作 记录。

栈顶记录指示当前层的执行情况，称之为当前活动记录。

递归工作栈的栈顶指针，称之为当前环境指针。

## 3.3 队列

### 队列的类型定义

队列(Queue)也是一种运算受限的线性表。它只允许在表的一 端进行插入，而在另一端进行删除。允许删除的一端称为队头 (front)，允许插入的一端称为队尾(rear)。

队列亦称作先进先 出(First In First Out)的线性表，简称FIFO表。

**双端队列**（deque）：

![image-20241215181624724](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241215181624724.png)

```
ADT Queue {

Status InitQueue(&Q)
操作结果：构造一个空队列 Q。

Status DestroyQueue(&Q)
初始条件：队列 Q 已存在。
操作结果：队列 Q 被销毁，不再存在。

Status ClearQueue(&Q)
初始条件：队列 Q 已存在。
操作结果：将 Q 清为空队列。

Status QueueEmpty(Q)
初始条件：队列 Q 已存在。
操作结果：若 Q 为空队列，则返回TRUE，否则返回FALSE。

Status QueueLength(Q)
初始条件：队列 Q 已存在。
操作结果：返回 Q 的元素个数，即队列的长度。

Status GetHead(Q,&e)
初始条件：Q 为非空队列。
操作结果：用 e 返回Q的队头元素。

Status EnQueue(&Q,e)
初始条件：队列 Q 已存在。
操作结果：插入元素 e 为 Q 的新的队尾元素。

Status DeQueue(&Q,&e)
初始条件：Q 为非空队列。
操作结果：删除 Q 的队头元素，并用 e 返回其值。

Status QueueTraverse(Q,visit( ))
初始条件：队列 Q 已存在且非空，visit( )为元素的访问函数。
操作结果：依次对 Q 的每个元素调用函数 visit()，一旦 visit( ) 失败则操作失败。
} ADT Queue
```

### 队列的存储表示和操作的实现

队列的链式存储结构简称为**链队列**，它是限制仅在表头删除和表尾插入的单链表。

为方便操作，为链队列增加头结点。**头结点**（Q.front）指向队头（**队头元素为a1**），并保存指向头结点的队头指针。

显然仅有单链表的头指针不便于在表尾做插入操作，为此再增加一个尾指针（Q.rear）， 指向链表的最后一个结点（队尾，元素为an）。

一个链队列包含**一个头指针和一个尾指针**。对于空队列，头尾指针相等。

![image-20241215181837584](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241215181837584.png)





### 循环队列

利用顺序分配存储结构实现队列时，需设立两个指针 front 和 rear 分别指示“队头”和“队尾”的位置。

为了叙述方便，在此约定：

- 初始化建空队列时，令 front=rear=0；
- 每当插入一个新的队尾元素后，尾指针 rear增1；
- 每当删除一个队头元素之后，头指针front增1

![image-20241215182550565](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241215182550565.png)

![image-20241215182603896](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241215182603896.png)

另设标志位以区别队列是“空”或“满” ：少用一个元素空间，以“队列头指针在队列尾指针的下一位置 上”作为队列“满” ；

队列空：Q.front == Q.rear；

队列满：Q.front == (Q.rear+1)%MAXSIZE；

除以上两种情况，队列内还有 (Q.rear-Q.front) 个元素。

![image-20241215183142962](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241215183142962.png)

```
#define MAX_QSIZE 100 /* 最大队列长度+1 */
typedef struct
{
	QElemType *base; /* 初始化的动态分配存储空间 */
	int front;
	int rear;
}SqQueue;

Status InitQueue (SqQueue &Q){
	Q.base = (QElemType*)malloc(MAXQSIZE*sizeof(QElemType));
	if(!Q.base) exit(OVERFLOW);
	Q.front=Q.rear=0;
	return OK;
}

Status EnQueue (SqQueue &Q, QElemType e){
// 若当前队列不满，就在当前队列的尾元素之后，
// 插入元素 e 为新的队列尾元素,并返回TRUE,否则返回FALSE。

	if((Q.rear+1) % MAXQSIZE == Q.front) return ERROR;
	Q.base[Q.rear] = e;
	Q = (Q.rear+1)%MAXQSIZE;
	return OK;
}

Status DeQueue (SqQueue &Q, QElemType &e)
{
// 若队列不空，则删除当前队列Q中的头元素，用 e 返回其值
// 并返回TRUE；否则返回 FALSE
	if (Q.front == Q.rear) return ERROR;
	e = Q.base[Q.front];
	Q.front = (Q.front+1) % MAXQSIZE;
	return OK;
}

int QueueLength (SqQueue Q){
// 返回队列Q中元素个数，即队列的长度
return ((Q.rear-Q.front+MAXQSIZE) % MAXQSIZE);
}
```



## 3.4 队列应用举例

##### 杨辉三角

## 3.5 离散事件模拟



# 第四章 串

## 4.1 串的类型定义 

#### 串的类型定义

**串(String)**是零个或多个字符组成的有限序列。一般记作 S=“a1a2a3…an ” ，其中S是串名，双引号括起来的字符序列是 串值；ai (1≦i≦n)可以是字母、数字或其它字符；串中所包含的字符个数称为该串的长度。长度为零的串称为空串(Empty String)，它不包含任何字符。

将仅由一个或多个空格组成的串称为**空格串(Blank String)**。

串中任意个连续字符组成的子序列称为该串的**子串**，包含子串 的串相应地称为**主串**。通常将子串在主串中首次出现时的**该子串的首字符对应的主串中的序号**，定义为**子串在主串中的序号** （或位置）。

特别地，空串是任意串的子串，任意串是其自身的子串。

通常在程序中使用的串可分为两种：串变量和串常量。

**串常量：**串常量和整常数、实常数一样，在程序中只能被引用但不能改变其值， 即只能读不能写。

**串变量**和其它类型的变量一样，其取值是可以改变的。

#### 串的基本操作

```
ADT String {
    基本操作：
    
    StrAssign (&T, chars)
    初始条件：chars 是串常量。
    操作结果：赋于串T的值为 chars。
    
    DestroyString (&S)
    初始条件：串 S 存在。
    操作结果：串 S 被销毁。
    
    StrCopy (&T, S)
    初始条件：串 S 存在。
    操作结果：由串 S 复制得串 T。
    
    StrEmpty (S)
    初始条件：串 S 存在。
    操作结果：若 S 为空串，则返回 TRUE，否则返回 FALSE。
    
    StrCompare (S, T)
    初始条件：串 S 和 T 存在。
    操作结果：若S>T，则返回值>0；若S=T，则返回值=0；若S<T，则返回值<0。
    
    StrLength (S)
    初始条件：串 S 存在。
    操作结果：返回串 S 序列中的字符个数，即串的长度。
    
    ClearString (&S)
    初始条件：串 S 存在。
    操作结果：将 S 清为空串。
    
    Concat (&T, S1, S2)
    初始条件：串 S1 和 S2 存在。
    操作结果：用 T 返回由 S1 和 S2 联接而成的新串。
    
    SubString (&Sub, S, pos, len)
    初始条件：串S存在，1≤pos≤StrLength(S)且0≤len≤StrLength(S)-pos+1。
    操作结果：用 Sub 返回串S的第 pos 个字符起长度为 len 的子串。
    
    Index (S, T, pos)
    初始条件：串S和T存在，T 是非空串，1≤pos≤StrLength(S)。
    操作结果：若主串S中存在和串T值相同的子串，则返回它在主串S中第pos个字符之后第一次出现的位置；否则函数值为0。
    
    Replace (&S, T, V)
    初始条件：串 S，T 和 V 存在，T 是非空串。
    操作结果：用V替换主串S中出现的所有与T相等的不重叠的子串。
    
    StrInsert (&S, pos, T)
    初始条件：串 S 和 T 存在，1≤pos≤StrLength(S)＋1。
    操作结果：在串 S 的第 pos 个字符之前插入串 T。
    
    StrDelete (&S, pos, len)
    初始条件：串 S 存在，1≤pos≤StrLength(S)-len+1。
    操作结果：从串 S 中删除第 pos 个字符起长度为 len 的子串。
} ADT String

int index(String S,String T,int pos){
// T为非空串。若主串S中第 pos 个字符之后存在与T相等的子串，
// 则返回第一个这样的子串在S中的位置，否则返回0。
    if (pos>0){
    	n=StrLength(S);
    	m=StrLength(T);
    	while(i<=n-m+1){
    		SubString(sub,S,i,m);
    		if(StrCompare(sub,T)!=0) ++i;
    		else return i;
    	}
    }
    return 0;
}

```

实现“置换”Replace (&S, T, V)操作的基本思想为：由S和V 生成一个新的串 news。首先将它初始化为一个空串，然后重复下列两步直至“查找不成功”为止：

- 1) 自起始位置起，在串S中查找和T相同的子串； 
- 2) 将S中不被置换的子串（即从 pos 起到和T相同子串在S中的 位置之前的字符序列）和V相继连接到news 串上。

```
void replace(String &S,String T,String V){
// 以串V替代串S中出现的所有和串T相同的子串
	n=StrLength(S);
	m=StrLength(T);
	pos=1;
	StrAssign(news,NullStr);
	i=1;
	while(pos<=n-m+1 && i){
	// 从pos指示位置起查找串T
		i=index(S,T,pos);
		if(i!=0){
			SubString(sub,S,pos,i-pos);
			Concat(news,news,sub);
			Concat(news,news,V);
			pos=i+m;
		}
	}
	SubString(sub,S,pos,n-pos+1);
	Concat(S,news,sub);
}
```

## 4.2 串的表示和实现 

#### 串有三种机内表示方法

##### 一、串的定长顺序存储表示

定长顺序存储表示,也称为静态存储分配的顺序表。它是用一组连续的存储单元来存放串中的字符序列。所谓定长顺序存储结构，是直接使用定长的字符数组来定义，数组的上界预先给出：

```
#define maxstrlen 255
typedef char sstring[maxstrlen+1];
sstring s;
```

**设置终结符**

一般可使用一个不会出现在串中的特殊字符在串值的尾部来表示 串的结束。例如，C语言中以字符‘\0’表示串值的终结，这就 是为什么在上述定义中，串空间最大值maxstrlen+1为256，但最 多只能存放255个字符的原因，因为必须留一个字节来存放‘\0’ 字符。

**不设终结符**

可用一个整数来表示串的长度。如PASCAL语言， 以下标为0的数组分量存储串的实际长度。



在这种表示方法下，实现串操作的基本操作是"**字符序列的复制**"。

```
// 算法 4.3 
// 以T返回由S1和S2联接而成的新串
Status Concat(SString &T,SString S1,SString S2){
	if(S1[0]+S2[0]<MAXSTRLEN){
		T[1..S1[0]] = S1[1..S1[0]];
		T[S1[0]+1..S1[0]+S2[0]] = S2[1..S2[0]];
		T[0] = S1[0] + S2[0]; 
		uncut = TRUE;
	}else if (S1[0]<MAXSTRLEN){
		T[1..S1[0]] = S1[1..S1[0]];
		T[S1[0]+1..MAXSTRLEN] = S2[1.. MAXSTRLEN-S1[0]];
		T[0] = MAXSTRLEN; uncut = FALSE;
	}else{
		T[0.. MAXSTRLEN] = S1[0.. MAXSTRLEN];
		uncut = FALSE; 
	}
	return uncut;
}
// 即在该程序中必须为T分配足够的空间，否则将会出现操作不成功。


// 算法 4.4
Status SubString (SString &Sub,SString S,int pos,int len){
	// 若参数合法(即1≤pos≤StrLength(S) 且
	// 0≤len≤StrLength(S)-pos+1),则以Sub带回串S中第pos个
	// 字符起长度为len的子串，并返回TRUE，否则返回FALSE
	if (pos < 1 || pos > S[0] || len < 0 || len > S[0] -pos+1) return FALSE;
	Sub[1...len] = S[pos..pos+len-1];
	Sub[0] = len;
	return TRUE;
}
```

##### 二、串的堆分配存储表示

仍以一组地址连续的存储单 元存放串值字符序列，但它们的存储空间是在程序执行过 程中动态分配而得，所以也称为动态存储分配的顺序表。

```
typedef struct {
    char *ch;
    int length;
} HString;

Status StrInsert(HString &S,int pos,Hstring T){
    // 若1≤pos≤StrLength(S)＋1，则改变串S，在串S的第pos个
    // 字符之前插入串T，并返回TRUE,否则串S不变，并返回FALSE
    if (pos < 1 || pos > S.length+1) return FALSE;
    if (T.length){
    	if(!(S.ch=(char*)realloc(S.ch,(S.length+T.length)*sizeof(char))) exit(OVERFLOW);
    	for(i=S.length-1;i>=pos-1;--i){
    		S.ch[i+T.length]=S.ch[i]
    	}
    	S.ch[pos-1..pos+T.length-2]=S.ch[0..T.length-1];
    	S.length+=T.length;
    }
}
```

##### 三、串的块链存储表示

- 采用链表方式存储串值
- 结点大小：每个结点可以存放一个字符，也可以存放多个 字符。
- 由于在一般情况下，串的操作都是从前往后进行的，因此 串的链表通常不设双链，也不设头结点，但为了便于进行 诸如串的联接等操作，链表中还附设有尾指针，并且由于 串的长度不一定是结点大小的整数倍（链表中最后一个结 点中的字符非都是有效字符），因此还需要一个指示串长 的域。

```
const CHUNKSIZE = 80; //块（结点）大小
typedef struct Chunk{
	char ch[CUNKSIZE];
	struct Chunk *next;
}Chunk;

typedef struct { // 串的链表结构
    Chunk *head, *tail; // 串的头指针和尾指针
    int curlen; // 串的当前长度
} LString;
```

由于串的结构的特殊性，即串的数据元素是一个字符，它只有8 位二进制数，因此用链表存储串值时通常采用的办法是在一个结 点中存放多个字符。图中的 “结点大小” CHUNKSIZE=4。

![image-20241216214319780](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241216214319780.png)

在以链表存储串值时，定义串的存储密度为

![image-20241216214426608](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241216214426608.png)

## 4.3 串的模式匹配算法 

#### 1、串的模式匹配的简单算法

```
int Index(SString S,SString T,int pos){
 	i=pos;
 	j=0;
 	while (i<S[0] && j<=T[0]){
 		if(S[i]==T[j]){++i;++j}
 		else{i=i+j-2;j=1}
 	}
 	if(j>T[0]) return i-T[0];//匹配成功
	else return 0; 
}
```

匹配成功时，函数返回的值是"子串在主串中的位置" ， 即子串中第1个字符在主串字符序列中的 "位置" ，而非该字符 在存储结构数组中的下标。

#### 2.KMP算法

KMP算法的核心思想是利用已经得到的部分匹配信息来进行后 面的匹配过程。

```
int KMP_Insert(SString S,SString T,int pos){
// 利用模式串T的next函数求T在主串S中第pos个字符之后第一次
// 出现的位置的KMP算法。其中1≤pos≤StrLength(S)
	i=pos;  //对应S
	j=1;	//对应T
	while(i<S[0] && j<T[0]){
		if(j==0 || S[i]==T[j]){
			++i;
			++j;
		}else{
			j=next[j];
		}
	}
	if(j>T[0]) return (i-T[0]);
	else return 0;
}
```

```
void get_next(SString T,int next[]){
	j=1;
	next[1]=0;
	k=0;
	while(j<T[0]){
		if(k==0 || T[j]==T[k]){
			++j;++k;
			next[j]=k;
		}else{
			k=next[k]
		}
	}
}
```

```
void get_nextval(){
// 求模式串T的next函数值并存入数组 next
	j=1;  //遍历串T
	next[1]=0;
	k=0;  //增长前后公共缀
	while(j<T[0]){
		if(k==0 || T[j]==T[k]){
			++j;++k;
			if(T[j]!=T[k]){
				next[j]=k;
			}else{
				next[j]=next[k];
			}
		}else{
			k=next[k];
		}
	}
}
```

## 4.4 串操作应用举例





# 第五章 数组和广义表

## 5.1 数组的类型定义 

```
ADT Array {
基本操作：
    InitArray (&A, n, bound1, ..., boundn)
    操作结果：若维数 n 和各维长度合法，则构造相应的数组 A。
    
    DestroyArray (&A)
    初始条件：数组 A 已经存在。
    操作结果：销毁数组 A。
    
    Value (A, &e, index1, ..., indexn)
    初始条件：A 是 n 维数组，e为元素变量，随后是 n 个下标值。
    操作结果：若各下标不超界，则e赋值为所指定的A的元素值，并返
    回OK。
    
    Assign (&A, e, index1, ..., indexn)
    初始条件：A 是n维数组，e 为元素变量，随后是 n 个下标值。
    操作结果：若下标不超界，则将 e 的值赋给A中指定下标的元素。
}ADT Array

```

## 5.2 数组的顺序表示和实现

![image-20241216224326395](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241216224326395.png) 

#### 行优先顺序

![image-20241216224403166](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241216224403166.png)

#### 列优先顺序

![image-20241216224353222](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241216224353222.png)

#### 存储地址

假设二维数组 R[m][n] 中每个数据元素占L个存储地址，并以 LOC(i, j) 表示 下标为 (i, j) 的数据元素的存储地址，则该数组中任何一对下标 (i, j) 对应的数 据元素。

在“以行为主”的顺序映象中的存储地址为：

![image-20241216224822123](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241216224822123.png)

 **LOC (i, j) = LOC(0,0) + (i×n+j) × L**

在“以列为主”的顺序映象中的存储地址为：

![image-20241216224921184](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241216224921184.png)

**LOC (i, j) = LOC(0,0) + (j×m+i) ×L**

类似地，假设三维数组 R[p] [m] [n] 中每个数据元素占 L 个存储地址，并 以 LOC(i,j,k) 表示下标为(i,j,k) 的数据元素的存储地址，则该数组中任何 一对下标(i,j,k) 对应的数据元素在“以行为主”的顺序映象中的存储地址为：

![image-20241216225207966](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241216225207966.png)

**LOC (i,j,k) = LOC(0,0,0) + (i×m×n + j×n+k)×L**

**推广到 N 维数组**，则得到

![image-20241216225258758](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241216225258758.png)

## 5.3 矩阵的压缩存储 

在数值分析的计算中经常出现一些有下列特性的高阶矩阵， 同时矩阵中**有很多值相同的元或零值元，为了节省存储空间**，需要对它们进行“压缩存储” ，即不存或少存这些值相同的元 或零值元。

#### 一、特殊矩阵的压缩存储方法

##### 1、对称矩阵

![image-20241216225940261](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241216225940261.png)

对称矩阵中的元素关于主对角线对称，故只要存储矩阵中上三角或 下三角中的元素，让每两个对称的元素共享一个存储空间，这样， 能节约近一半的存储空间。不失一般性，我们按“行优先顺序”存 储主对角线（包括对角线）以下的元素，其存储形式如下图所示：

![image-20241216230102494](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241216230102494.png)

**计算k：元素存储的个数**

若i≥j，则aij在下三角形中。

**k=i×(i+1)/2+j    0≦k<n(n+1)/2**

若i<j，则aij在上三角形中。

**k=j× (j+1)/2+i    0≦ k<n(n+1)/2**

令 I=max(i,j)， J=min(i,j)，则k和 i, j的对应关系可统一为：

**k=I× (I+1)/2+J    0≦ k<n(n+1)/2**

aij的地址可用下列式计算：

![image-20241216231436147](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241216231436147.png)

##### 2、三角矩阵

以主对角线划分，三角矩阵有上三角和下三角两种。上三角矩阵 如图(a)所示，它的下三角（不包括主对角线）中的元素均为常数。下 三角矩阵正好相反，它的主对角线上方均为常数，如图(b)所示。在大 多数情况下，三角矩阵常数为零。

![image-20241216231639204](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241216231639204.png)

三角矩阵中的重复元素c可共享一个存储空间，其余的元素正好有n(n+1)/2 个，因此，三角矩阵可压缩存储到向量sa[0..n(n+1)/2]中，其中**c存放在向量的最后一个分量中。**

![image-20241216232403508](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241216232403508.png)

##### 3、对角矩阵

除了主对角线和主对角线相邻两侧的若干条对角线上的元素之外， 其余元素皆为零。显 然，当∣i-j∣>1时，元素aij=0。

![image-20241216232505737](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241216232505737.png)

对角矩阵可按行优先顺序或对角线的顺序，将其压缩存储到一个向量中， 并且也能找到每个非零元素和向量下标的对应关系。 

例如：在3对角矩阵里除满足条件i=0，j=0、1，或i=n-1，j=n-2、n-1 或1 ≦ i<n-1,j=i-1、i、i+1的元素aij以外，其余元素为0。

对这种矩阵，我们也可按行优序为主序来存储。除第0行和第n-1行是 2个元素外，每行的非零元素都要是3个，因此，需存储的元素个数为 3n-2。

![image-20241216233453474](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241216233453474.png)

![image-20241216233925585](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241216233925585.png)

#### 二、稀疏矩阵的压缩存储方法

如果矩阵中只有少量的非零值元，并且这些非零值元在矩阵中的分布 没有一定规律，则称为随机稀疏矩阵，简称为稀疏矩阵。

![image-20241216234114794](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241216234114794.png)

压缩存储的基本宗旨是只存放矩阵中的非零值元，则在存储非零元的值的同时必须记下它在矩阵中的位置(i, j) ，反之，一个三元组(i, j, aij)唯一确定了矩阵A中的一个非零值元，由此可以用“数据元素为上述三元组的线性表”表示稀疏矩阵，并且非零元在三元组线性表中是“以行为主”有序排列的。

##### 1、三元组顺序表

![image-20241216234232697](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241216234232697.png)

```
const MAXSIZE=12500; // 假设非零元个数的最大值为12500
typedef struct{
	int i,j;
	ElemType e;   // 该非零元的值
}Triple;

typedef struct{
	Triple data[MAXMIZE+1];   // 非零元三元组表，data[0]未用
	int mu,nu,tu;   // 矩阵的行数、列数和非零元的个数
}TSMatrix;
```

<img src="C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241216235007376.png" alt="image-20241216235007376" style="zoom:80%;" />

**转置运算**

对比M和T，**T的行数、列数和非零元的个数**等于**M的列数、行数和 非零元的个数**，且T中的每个非零元和M中的非零元相比，它们的值相同，但**行列号互换**。

三元组在M中按行优先顺序排列，在T中，由于行列交换操作，因此三元组在T中按列优先顺序 排列。

```
Status TransposeSMatrix(TSMatrix M,TSMatrix &T){
	//将稀疏表示的矩阵M转置，生成矩阵T
	T.mu=M.nu;
	T.nu=M.mu;
	T.tu=M.tu;
	if(T.tu){
		q=1; //初始化三元组顺序表（T.data）索引
		for(col=1;col<M.nu;++col){
			for(p=1;p<M.mu;++p){
				if(M.data[p].j == col){
					T.data[q].i=M.data[p].j;
					T.data[q].j=M.data[p].i;
					T.data[q].e=M.data[p].e;
					++q;
				}
			}
		}
	}
	return OK;
}
```

**快速转置**

![image-20241217000944940](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241217000944940.png)

![image-20241217001322984](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241217001322984.png)



![image-20241217002255188](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241217002255188.png)

```
Status FastTransposeSMatrix(TSMatrix M,TSMatrix &T){
// 采用三元组顺序表存储表示，求稀疏矩阵M的转置矩阵 T
	T.mu=M.nu;
	T.nu=M.mu;
	T.tu=M.tu;
	if(T.tu){
		for(col=1;col<=M.nu;++col){
			num[col]=0;
		}
		for(t=1;t<M.mu;t++){
			++num[M.data[t].j];
		}
		cpot[1]=1;
		for(col=2;col<M.nu;++col){
			cpot[col]=copt[col-1]+num[col-1];
		}
		// 求T中每一行的第一个非零元在T.data中的序号
		for(p=1;p<=M.tu;++p){
			col=M.data[p].j;
			q=cpot[col];
			T.data[q].j=M.data[p].i;
			T.data[q].i=M.data[p].j;
			T.data[q].e=M.data[p].e;
			++cpot[col];
		}
	}
	return OK;
}
```

上述算法的时间复杂度为O (M.nu+M.tu)。 如果tu和mu×nu等数量级， 则时间复杂度为O(mu×nu)

##### 2、行逻辑链接的顺序表

为了便于随机存取任意一行的非零元，需要知道每行首个非零元素所在位置。可将 cpot 中的值视作指向每行第一个非零元的“”指针，故称这种表示方法 为“行逻辑链接”的顺序表。

```
// 行逻辑链接的三元组顺序表的结构定义
const MAXRC=500;	// 矩阵行数和列数的最大值
const MAXSIZE=12500; 	// 假设非零元个数的最大值为12500
typedef struct {
	Triple data[MAXSIZE + 1]; //非零元三元组表，data[0] 未用
	int rpos[MAXRC + 1]; // 指示各行第一个非零元在data中的位置
	int mu, nu, tu; // 矩阵的行数、列数和非零元的个数
}RLSMatrix; // 行逻辑链接顺序表

```



##### 3、十字链表



## 5.4 广义表的定义 

**广义表**（Lists，又称列表）是线性表的推广。

**广义表**是n(n>=0)个元素a1,a2,a3,…,an的有限序列，其中ai或者是 原子项，或者是一个广义表。通常记作LS=（a1,a2,a3,…,an)。LS 是广义表的名字，n为它的长度。若ai是广义表，则称它为LS的子 表。

广义表LS=（a1,a2,a3,…,an)兼有线性结构和层次结构 的特性，归纳如下：

- 数据元素有固定的相对次序；
- 长度定义为最外层括弧中包含的数据元素个数；例如广义表E=(a, (b,c))，及其子表（b,c）的长度都是2。
- 深度定义为广义表书写形式中括弧的最大重数； 例如广义表E=(a, (b,c))，D=((),(e),(a,(b,c,d)))的深度分别是2和3。
- 广义表可被其它广义表所共享；
- \5. 广义表可以是一个递归的表，递归表的深度可以是无穷，但长度有限；例如广义表E=(a, E)=(a,(a,(a,(…)))) 的长度都是2，深度是无穷。
- 对于任意一个非空广义表LS=（a1,a2,a3,…,an) ，它的第一个数据元素被 定义为广义表的“表头”，Head(LS)=a1，而由其余数据元素构成的广义表 被 定义为广义表的“表尾”，Tail(LS)= (a2,a3,…,an) 。

```
ADT GList {
基本操作：

{结构初始化}
InitGList(&L);
操作结果：创建空的广义表 L。

CreateGList(&L, S);
初始条件：S是广义表的书写形式串。
操作结果：由S创建广义表 L。

CopyGList(&T, L);
初始条件：广义表 L 存在。
操作结果：由广义表 L 复制得到广义表 T。


{销毁结构}
DestroyGList(&L);
初始条件：广义表 L 存在。
操作结果：销毁广义表 L。


{引用型操作}

GListLength(L);
初始条件：广义表 L 存在。
操作结果：求广义表 L 的长度，即元素个数。

GListDepth(L);
初始条件：广义表 L 存在。
操作结果：求广义表 L 的深度。

GListEmpty(L);
初始条件：广义表 L 存在。
操作结果：判定广义表 L 是否为空表。

GetHead(L);
初始条件：广义表 L 存在且非空。
操作结果：返回广义表 L 的表头。

GetTail(L);
初始条件：广义表 L 存在且非空。
操作结果：返回广义表 L 的表尾。


{加工型操作}

InsertFirst_GL(&L, e);
初始条件：广义表 L 存在。
操作结果：插入元素 e 作为广义表 L 的第一个元素。

DeleteFirst_GL(&L, &e);
初始条件：广义表 L 存在。
操作结果：删除广义表 L 中第一个元素，并用 e 返回其值。

Traverse_GL(L, Visit());
初始条件：广义表 L 存在。
操作结果：遍历广义表 L，用函数 Visit 处理每个元素。
} ADT GList
```

## 5.5 广义表的存储结构

表示广义表的链表中的结点为“异构”结点，结点中设有一个“标志域tag”， 并约定 tag=0 表示原子结点，tag=1表示表结点。

原子结点中的 data 域存储原子，表结点中指针域的两个值分别指向表头和表尾。 

空表：ls=Nil

非空表：ls指向表节点

```
typedef enum{
	ATOM=0;
	LIST=1;
}ElemTag;

typedef struct GLNode{
	ElemTag tag;
	union{
		AtomType data;// data是原子结点的值域，AtomType由用户定义
		struct{ 
			struct GLNode *hp,*tp;
		}ptr;
		// ptr是表结点的指针域，ptr.hp和ptr.tp分别指向表头和表尾
	}
}*GList;
```





# 第六章 树和二叉树

## 6.1 树的定义和基本术语 

**定义**：树(Tree)是n(n>=0)个结点的有限集T，它满足如下两个 条件：

- (1)有且仅有一个特定的称为根(Root)的结点；
- (2) 其余的结点可分为 m(m>=0) 个 互不相交的子集 T1,T2,T3…Tm，其中每个子集又是一棵树，并称其为子树 (Subtree)。

T为空时称为空树。

### 树结构基本术语：

- **分支**：称根和子树根之间的连线为“分支” ；
- **结点**：数据元素和所有指向其子树根的分支构成树中一个“结点” ；
- **结点的度**：结点的分支的个数定义为“结点的度” ，如结点B的度为2，D 的 度为3。 
- **树的度**：树中所有结点度的最大值定义为“树的度”。如下例，树的度为3.
- **叶子、分支结点**：称度为零的结点为“叶子”或“终端结点”，如结点 K,L,G 等， 反之所有度不为零的结点被称作“分支结点”或“非终端结点" ，如结点 F,J 等。
- 树是一种层次分明的结构，约定根的层次为1，根的孩子为第二层。 若某节点的层次为L，则其子树的根的层次为 L+1。
- **树的深度**定义为树中结点的最大层次，也称为树的高度。
- 子树之间可能存在两种情况，如果子树之间映射客观存在次序关系， 则为**“有序树”** ，否则为 **“无序树”**。
- 同一双亲的孩子之间互为**“兄弟”**。双亲在同一层的节点互为**“堂兄弟”**。
- 结点的**“祖先”**指从根结点到该结点所经分支上的**所有结点**。
- 以某结点为根的子树中的任一结点都称为该结点的**“子孙”**。

**森林：**为 m(m≥0) 棵互不相交的树的集合。则对树中每 个结点而言，其子树的集合即为森林。

Tree = ( root，F)，其中，root 是数据元素，称作树的根， F是**子树森林**，记作F=( T1 ,T2 ,…,Tm )，其中 Ti=(ri，Fi ) 称 作根 root 的第 i 棵子树，当 m≠0 时，在树根和其子树森林 之间存在下列关系： 

RF={  | i=1,2,…,m, m>0}

```
ADT Tree {
基本操作：


{结构初始化}
InitTree(&T);
操作结果：构造空树 T。

CreateTree(&T,definition);
初始条件：definition 给出树T的定义。
操作结果：按 definition 构造树 T。

{销毁结构}
DestroyTree(&T);
初始条件：树 T 存在。
操作结果：销毁树 T。


{引用型操作}
TreeEmpty(T);
初始条件：树 T 存在。
操作结果：若 T 为空树，则返回 TRUE，否则返回 FALSE。


{引用型操作}
TreeDepth(T);
初始条件：树T存在。
操作结果：返回T的深度。//树的深度定义为树中叶子结点所在最大层次数

Root(T);
初始条件：树 T 存在。
操作结果：返回 T 的根。

Value(T, cur_e);
初始条件：树 T 存在，cur_e 是 T 中某个结点。
操作结果：返回 cur_e 的值。

Parent(T, cur_e);
初始条件：树 T 存在，cur_e 是 T 中某个结点。
操作结果：若 cur_e 是T的非根结点，则返回它的双亲，否则返回“空”。

LeftChild(T, cur_e);
初始条件：树 T 存在，cur_e 是 T 中某个结点。
操作结果：若 cur_e 是T的非叶子结点，则返回它的最左孩子，否则返回“空”。


{引用型操作}
RightSibling(T, cur_e);
初始条件：树 T 存在，cur_e 是 T 中某个结点。
操作结果：若 cur_e 有右兄弟，则返回它的右兄弟，否则返回“空”。

TraverseTree(T, visit());
初始条件：树T存在，visit 是对结点操作的应用函数。
操作结果：按某种次序对 T 的每个结点调用函数 visit() 一次且至多一次。一旦 visit() 失败，则操作失败。


{加工型操作}
Assign(T, cur_e, value);
初始条件：树T存在，cur_e 是 T 中某个结点。
操作结果：结点 cur_e 赋值为 value。

ClearTree(&T);
初始条件：树 T 存在。
操作结果：将树 T 清为空树。


{加工型操作}
InsertChild(&T, &p, i, c);
初始条件：树 T 存在，p 指向T中某个结点，1≤i≤p 所指结点的度＋1，非空树 c 与 T 不相交。
操作结果：插入 c 为 T 中 p 所指结点的第 i 棵子树。

DeleteChild(&T, &p, i);
初始条件：树 T 存在，p 指向 T 中某个结点，1≤i≤p 指结点的度。
操作结果：删除 T 中 p 所指结点的第 i 棵子树。

} ADT Tree
```

### 树的其它表示形式

树的几种表示法：树形图表示，嵌套集合表示法，凹入表表示法，广义表表示法

![image-20241226225226899](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241226225226899.png)

## 6.2 二叉树 

### 1、二叉树(Binary Tree)的定义

二叉树是由n(n>=0)个结点的有限集合构成，此集合或者为空集， 或者由一个根结点及两棵互不相交的左、右子树组成，并且左右子 树都是二叉树。

###  2、二叉树(Binary Tree)的特点

每个结点至多只有两棵子树（二叉树中不存在度大于2的结点）；二叉树的子数有左右之分，次序不能任意颠倒。

### 3.二叉树的性质

![image-20241217210126148](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241217210126148.png)

![image-20241217210158200](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241217210158200.png)

![image-20241217210721450](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241217210721450.png)

#### 满二叉树和完全二叉树

一棵深度为k且由2 k -1个结点的二叉树称为**满二叉树**。

如果深度为k、由n个结点的二叉树中，当且仅当其每一个结点能 够与深度为k的顺序编号的满二叉树从1到n标号的结点相对应，则 称这样的二叉树为**完全二叉树**。

满二叉树是完全二叉树的特例。

![image-20241217210926666](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241217210926666.png)

![image-20241217210946853](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241217210946853.png)

![image-20241217211034709](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241217211034709.png)

![image-20241217211344163](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241217211344163.png)

### 4.二叉树——存储结构

#### (1) 顺序存储结构

是用一组连续的存储单元存储二叉树的数据元素。

从树根起，自上层至下层，每层自左至右的给所有结点编号的缺点是：有可能对存储空间造成极大的浪费。

```
# define MAX_TREE_SIZE 100;
typedef TElemType
SqBiTree[MAX_TREE_SIZE];
SqBiTree bt;
```

#### (2) 二叉链表法

![image-20241217211756466](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241217211756466.png)

```
typedef struct BiTNode{
	TElemType data;
	struct BiTNode *lchild,*rchild;
}BiTNode,*BiTree;
```

#### (3) 三叉链表法

```
typedef struct TriTNode{
	TElemType data;
	struct TriTNode *lchild,*rchild,*parent;
}*TriTree;
```

## 6.3 遍历二叉树和线索二叉树 

### 遍历二叉树

DLR——先（根）序遍历，

LDR——中（根）序遍历，

LRD——后（根）序遍历.

#### 遍历二叉树——前序遍历递归算法1

```
Status PreOrderTraverse(BiTree T,Status(*visit)(TElemType e))
{// 采用用二叉链表存储结构，Visit是对数据元素操作的应用函数，先序遍历
以T为根指针的二叉树的递归算法，对每个数据元素调用函数Visit。

	if(T){
		if(Visit(T->data)){
		// 先序遍历左子树
			if(PreOrderTraverse(T->Lchild,Visit)){
			if(PreOrderTraverse(T->Rchild,Visit)) return OK;
			}
		}
		return ERROR;
	}else return OK;
}
```

#### 遍历二叉树——中序遍历递归算法2

```
Status InOrderTraverse(BiTree T,Status(*visit)(TElemType e))
{// 采用用二叉链表存储结构，Visit是对数据元素操作的应用函数，中序遍历
以T为根指针的二叉树的递归算法，对每个数据元素调用函数Visit。

	if(T){
		if(InOrderTraverse(T->Lchild,Visit)){
		// 先序遍历左子树
			if{(Visit(T->data)){
				if(InOrderTraverse(T->Rchild,Visit)) return OK;
			}
		}
		return ERROR;
	}else return OK;
}
```

#### 遍历二叉树——中序遍历非递归算法1

```
Status InOrderTraverse (BiTree T, Status (* Visit)(TElemType e)){
	InitStack(S);
	Push(S,T);
	while(!StackEmpty(S)){
		while(GetTop(S,p) && p){
			Push(S,p->lchild);
		}
		Pop(S,p);
		if(!StackEmpty(S)){
			Pop(S,p);
			if(!visit(p->data)) return NULL;
			Push(S,p->rchild);
		}
	}
	return OK;
}
```

#### 遍历二叉树——中序遍历非递归算法2

```
Status InOrderTraverse (BiTree T, Status (* Visit)(TElemType e)){
// 采用二叉链表存储结构，Visit是对数据元素操作的应用函数。
// 中序遍历二叉树T的非递归算法，对每个数据元素调用函数Visit。
	InitStack(S);
	p=T;
	whilee (p || !StackEmpty(S)){
		if(p){
			Push(S,p);
			p=p->lchild;
		}else{
			Pop(S,p);
			p=p->rchild;
		}
	}
	return OK;
}
```

#### 遍历二叉树的应用

**统计二叉树叶子结点的个数（先序遍历）；**

```
void CountLeaf(BiTree T,int &count){
	if(T){
		if(!T->lchile && !T->rchild) count++;
		CountLeaf(T->lchild,count);
		CountLeaf(T->rchild,count);
	}
}
```

**求二叉树的深度（后序遍历）**

```
int Depth(BiTree T){
	if(!T){
		depthval=0;
	}else{
		depthLeft=Depth(T->lchild);
		depthRight=Depth(T->lchild);
		depthval=1+(depthleft>depthright?depthleft:depthright);	
	}
	return depthval;
}
```

**复制二叉树（后序遍历）**

```
BiTNode *GetTreeNode(TElemType item,BiTNode,*lptr,BiTNode *rptr){
//more about create
	if(!(BiTnode *)malloc(sizeof(BiTNode *))) exit(1);
	T->data = item;
    T->lchild = lptr;
    T->rchild = rptr;
    return T;
}

BiTNode *CopyTree(BiTNode *T)
{
    if( !T) return NULL;
    if( T->lchild )
         newlptr = CopyTree(T->lchild);
    else newlptr = NULL;
    if( T->rchild )
    	newrptr = CopyTree(T->rchild);
    else newrptr = NULL;
    	newnode = GetTreeNode(T->data, newlptr, newrptr);
    return newnode;
}

Status CreateBiTree(BiTree &T){
// 在先序遍历二叉树的过程中输入二叉树的“先序字符串”
，
// 建立根指针为 T的二叉链表存储结构。在先序字符串中，
// 字符‘#’表示空树，其它字母字符为结点的数据元素
	scanf(&ch);
	if (ch==‘#’) T=NULL; // 建空树
	else {
		if (!(T = (BiTNode *) malloc(sizeof(BiTNode)))) exit(OVERFLOW);
		T->data = ch; // “访问”操作为生成根结点
		CreateBiTree(T->Lchild); // 递归建(遍历)左子树
		CreateBiTree(T->Rchild); // 递归建(遍历)右子树
	}
}
```

**表达式字符串建二叉树**

按原表达式建二叉链表算法的基本思想为：

```
if (当前识别的字符 ch 是操作数)
    { 建叶子结点; 暂存; }
else if (当前识别的字符 ch 是运算符)
    {
    和前一个运算符比较优先数;
    若当前的优先数"高"，则暂存;
    否则以前一运算符为根建立一棵新的子树
    (最近建立的两棵子树分别为它的左右子树)并暂存;
}
在这个算法中需要两个栈，一个是运算符栈，其作用和表达式转换为后缀式的算法中相同，另一个是暂存“已经建好的子树根的指针”栈。
```

### 线索二叉树(*)

遍历二叉树的结果是求得结点的一个线性序列。 

当以二叉链表作为存储结构时，只能找到结点的左右孩子的信息，而 不能在结点的任一序列的前驱与后继信息，这种信息只有在遍历的动态 过程中才能得到，为了能保存所需的信息，可增加标志域：

![image-20241218174901445](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241218174901445.png)

以这种结构构成的二叉链表作为二叉树的存储结构，叫做**线索链表**，其 中指向结点前驱与后继的指针叫**做线索**。加上线索的二叉树称之为线索 二叉树。

在线索链表中添加了一个“头结点” ，头结点的 左指针指向二叉树的根结点，其右线索指向中序序列中的最后 一个结点(中序遍历中最后访问的一个节点)，中序序列中的第一个结点的左线索和中序序列中的 最后一个结点的右线索均指向头结点。空二叉树的线索链表的 头结点的左右指针回指到头结点。

![image-20241218175850617](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241218175850617.png)

**二叉树的二叉线索链表存储表示**

```
typedef enum PointerTpye{
	Link=0,Thread=1;
}
typedef struct BiThrNode{
    TElemType data;
    struct BiThrNode *Lchild, *Rchild; // 左右指针
    PointerTag LTag, RTag; // 左右指针类型标志
} BiThrNode, *BiThrTree;
```

如何在中序线索链表上进行遍历：

- 找到第一个结点：中序遍历访问的第一个节点必定“左子树为空”，若根结点没有左子树，则根结点即为中序遍历访问的第一个结点；即从根结点出发，顺指针 Lchild 找到其左子树直至某个结点的指针 Lchild 为“线索”止。
- 如何在中序线索链表中找结点的后继：若结点有右子树，则它的后继应该是其右子树中访问的第一个结点。该从它的右子树根出发，顺指针 Lchild 直至没 有左子树的结点为止，该结点即为它的后继。
- 图所示例子中的“结点D”为中序序列中的第一个结点
- 图中结点B的后继为结点G。
- ![image-20241218180613082](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241218180613082.png)

```
Status InOrderThreading(BiThrTree &Thrt, BiThrTree T){// T 指向中序线索链表中的头结点，
头结点的左指针 Lchild 指向二叉树的根结点，
头结点的右线索 Rchild 指向中序遍历访问的最后一个结点。
本算法对此二叉树进行中序遍历，对树中每个数据元素调用函数 Visit 进行访问操作。
	p=T->Lchild;
	while(p!=T){			// 空树或遍历结束时，p==T
		while(p->LTag==Link){  //存在左子树
			p=p->Lchild;
		}
		//现在p所在的节点左子树为空
		if(! Visit(p->data)) return ERROR;
		// 访问“右线索”所指后继结点
		while(p->RTag==Thread && p->Rchild!=T){
			p=p->Rchild;
			Visit(p->data);
		}
		p = p->Rchild;
	}
	return OK;
}
```

**如何建立二叉树的线索链表**

在遍历过程中建立，即在遍历过程中改变二叉链表中结点的 “空指针”以及相应的“指针类型标志”。

- 若结点没有左子树，则令其左指针指向它的“前驱” ,左指针类 型标志 “Thread”；
- 若结点没有右子树，则令它的右指针指向它的“后继” ,右指针 类型标志 “Thread”。
- 为了获取“前驱”的信息，需要在遍历过程中添加一个指向其前 驱的指针 pre。

```
Status InOrderThreading(BiThrTree &Thrt, BiThrTree T){
// T为指向二叉树根结点的指针，由此二叉链表建立二叉树
// 的中序线索链表，Thrt 指向线索链表中的头结点。

	if (!(Thrt = (BiThrTree)malloc(sizeof(BiThrNode)))) exit (OVERFLOW);
	Thrt->LTag = Link; 
	Thrt->RTag =Thread;
	Thrt->Rchild = Thrt; 
	if (!T) Thrt->Lchild = Thrt;
	else{
		Thrt->Lchild=T;
		pre=Thrt;
		InThreading(T); 
		pre->Rchild = Thrt; 
		pre->RTag = Thead;
		Thrt->Rchild = pre;
	}
}
```

```
void InThreading(BiThrTree p){
// 对 p 指向根结点的二叉树进行中序遍历，遍历过程中进行“中序线索化”。若p所指结点的左指针为空，则将它改为“左线索”，若 pre 所指结点的右指针为空，则将它改为“右线索”。指针 pre 在遍历过程中紧随其后，即始终指向 p所指结点在中序序列中的前驱。
	if(p){
		InThreading(p->Lchild);
		if (!p->Lchild){
			p->LTag = Thread; 
			p->Lchild = pre;
		} 
		if(!p->Rchild){
			pre=p;
			InThreading(p->Rchild);
		}
	}
}
```

## 6.4 树和森林 

### 树和森林－双亲表示法

每个结点中附设一个指示 器指示其双亲结点在链表中的位置

```
＃define MAX_TREE_SIZE 100
// 双亲表示法节点
typedef struct PTNode{
	TElemType data;
	int parent;
}PTNode;

typedef struct{
	PTNode nodes[MAX_TREE_SIZE];
	int r,n; //根的位置和节点数
}PTree
```

![image-20241218215129313](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241218215129313.png)

![image-20241218215143600](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241218215143600.png)

### 树和森林－孩子表示法

![image-20241218215710537](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241218215710537.png)

- 1.把每个结点的孩子视为一个线性表。结点的度不同，因此采用链式存储结构。
-  2.N个结点则有N个链表，它们的头指针又组成一个线性表。为方便访问，采用顺序 存储结构。

```
typedef struct CTNode{  // 孩子结点
	int child;
	struct CTNode *next;
}*ChildPtr;

// 双亲结点结构
typedef struct{
	ElemType data;
	ChildPtr firstchild;
}CTBox;

// 树结构
typedef struct{
	CTBox nodes[MAX_TREE_SIZE];
	int n,r;
}
```

### 树和森林－二叉链表（孩子兄弟）表示法

```
typedef struct CSNode{
    ElemType data;
    struct CSNode *firstchild, *nextsibling;
} CSNode, *CSTree;
```

- 其一，firstchild 指向该结点的“第一个”子树根结点；
- 其二，nextsibling 指向它的“下一个”兄弟结点。

![image-20241218220339634](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241218220339634.png)

（左边是孩子，右边是兄弟）

### 森林和二叉树的转换

![image-20241218220558378](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241218220558378.png)

![image-20241218221048574](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241218221048574.png)

![image-20241218220727328](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241218220727328.png)

### 树和森林的遍历

和二叉树的遍历类似，树的遍历问题亦为，从根结点出发，对树中各个结点进行一次且仅进行一次访问。 

对树进行遍历可有两条搜索路径：一条是从左到右（这里的左右指的是 在存储结构中自然形成的子树之间的次序），另一条是按层次从上到下。

#### 树的遍历

##### 一、先根(次序)遍历树

若树不空，则先访问根结点，然后依次从左到右先根遍历根的各棵子树。

##### 二、后根(次序)遍历树

若树不空，则先依次从左到右后根遍历根的各棵子树，然后访问根结点；

#### 森林的遍历

##### 一、先序遍历森林

![image-20241218221543867](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241218221543867.png)

##### 二、中序遍历森林

![image-20241218221555392](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241218221555392.png)

## 6.5 赫夫曼树及其应用

#### 赫夫曼树

赫夫曼（Huffman）树，又称最优树，是一类带权路径长度最短的树。

路径：从**树中一个结点到另一个结点之间的分支**构成这两个结点之间的路径；

路径长度：路径上的**分支数目**叫路径长度。

结点的路径长度：从**根结点**到该结点的路径上分支的数目。

**树的路径长度指**的是从树根到树中其余每个结点的路径长度之和。完全二叉树就是这种路径长度最短的二叉树。 

**结点的带权路径长度**则定义为从树根到该结点之间的路径长度与该结点上所带权值的乘积。假设树上有 n 个叶子结点，且每个叶子结点上带有权值 为 Wk(k=1,2,…,n)，则树的带权路径长度定义为树中所有叶子结点的带权 路径长度之和，通常记作 ：

![image-20241218222203251](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241218222203251.png)

其中： 为带权wk的叶子结点的路径长度。

![image-20241218222306272](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241218222306272.png)

##### 最优树的构造方法

![image-20241218222424441](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241218222424441.png)

##### 最优前缀编码

###### 等长编码

这类编码的二进制串的长度取决于电文中不同 的字符个数，假设需传送的电文中只有四种字符，只需两位字符 的串便可分辨，但如果电文中可能出现26种不同字符，则等长编 码串的长度为5。

###### 不等长编码

在实用的不等长编码中，任意一个字符的编码都不能是另一 个字符的编码的前缀，这种编码称为前缀编码。

![image-20241218223014903](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241218223014903.png)

```
typedef struct {
	unsigned int weight;
	unsigned int parent,lchild,rchild;
}HTNode,*HuffmanTree;/* 动态分配数组存储赫夫曼树 */
typedef char **HuffmanCode; /* 动态分配数组存储赫夫曼编码表 */
```







# 第七章 图

## 7.1 图的定义和术语 

### 图的定义和术语

#### 有向图

![image-20241218224504751](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241218224504751.png)

#### 无向图

![image-20241218224537190](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241218224537190.png)

#### 基本操作

```
ADT Graph {
{结构初始化}

CreateGraph(&G, V, VR);
初始条件：V 是图的顶点集，VR 是图中弧的集合。
操作结果：按 V 和 VR 的定义构造图 G。

{销毁结构}
DestroyGraph(&G);
初始条件：图 G 存在。
操作结果：销毁图 G。

{引用型操作}
LocateVex(G, u);
初始条件：图 G 存在，u 和 G 中顶点有相同特征。
操作结果：若 G 中存在和 u 相同的顶点，则返回该顶点在图中位置；
否则返回其它信息。

GetVex(G, v);
初始条件：图 G 存在，v 是 G 中某个顶点。
操作结果：返回 v 的值。

FirstAdjVex(G, v);
初始条件：图 G 存在，v 是 G 中某个顶点。
操作结果：返回 v 的第一个邻接点。若该顶点在 G 中没有邻接点，则返回“空”。

{引用型操作}
NextAdjVex(G, v, w);
初始条件：图 G 存在，v 是 G 中某个顶点，w 是 v 的邻接顶点。
操作结果：返回 v 的（相对于 w 的）下一个邻接点。若 w 是 v 的最后一个邻接点，则返回“空”。

{加工型操作}
PutVex(&G, v, value);
初始条件：图 G 存在，v 是 G 中某个顶点。
操作结果：对 v 赋值 value。
InsertVex(&G, v);
初始条件：图 G 存在，v 和图中顶点有相同特征。
操作结果：在图 G 中增添新顶点 v。

DeleteVex(&G, v);
初始条件：图 G 存在，v 是 G 中某个顶点。
操作结果：删除 G 中顶点 v 及其相关的弧。

{加工型操作}
InsertArc(&G, v, w);
初始条件：图 G 存在，v 和 w 是 G 中两个顶点。
操作结果：在G中增添弧<v,w>，若G是无向的，则还增添对称弧<w,v>。

DeleteArc(&G, v, w);
初始条件：图 G 存在，v 和 w 是 G 中两个顶点。
操作结果：在G中删除弧<v,w>，若G是无向的，则还删除对称弧<w,v>。

DFSTraverse(G, Visit());
初始条件：图 G 存在，Visit 是顶点的应用函数。
操作结果：对图G进行深度优先遍历。遍历过程中对每个顶点调用函数Visit 一次且仅一次。一旦 visit() 失败，则操作失败。

BFSTraverse(G, Visit());
初始条件：图 G 存在，Visit 是顶点的应用函数。
操作结果：对图G进行广度优先遍历。遍历过程中对每个顶点调用函数Visit 一次且仅一次。一旦 visit() 失败，则操作失败。

} ADT Graph
```

#### 完全图

有 n(n-1)/2 条边的无向图。其中 n 是结点个数。（两两之间存在一条连线）

#### 有向完全图

有 n(n-1) 条边的有向图。其中 n 是结点个数。（两两之间存在一条连线）

#### 稀疏图/稠密图

有很少的边或弧（e<nlogn）的图称为稀疏图，反之成为稠密图。

#### 邻接点：

无向图G=(V,{E})，如果边(V,V’)∈E，则称V和V’互为邻接 点。边(V,V’)和顶点V，V’相关联。

### 顶点的度：

对无向图而言，与一个顶点相关联的边的数目，或者说与其相邻接的顶点数为该顶点的度（TD）。例如左下方无向图中顶点B的度为3，顶点C的度为2。

对有向图而言，顶点的度为其出度和入度之和，其中**出度**（OD）定义为以**该顶点 为弧尾**的弧的数目，**入度**（ID）定义为以**该顶点为弧头**的弧的数目。例如右下方 有向图中顶点D的度为3，其中出度为2，入度为1，顶点B的度为3，其中出度为1，入度为2。

![image-20241218230234262](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241218230234262.png)

### 子图

![image-20241218230543799](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241218230543799.png)

### 路径和回路

![image-20241218230609947](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241218230609947.png)

![image-20241218230719738](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241218230719738.png)

### 连通图和连通分量

无向图：若无向图中任意两个顶点之间都存在一条无向路径，则称该无向图 为**连通图**，否则称为非连通图。

有向图：若有向图中任意两个顶点之间都存在一条有向路径，则称该有向图为**强连通图**。

**非连通图**中各个**极大连通子图**称作该图的**连通分量**。

非强连通的有向图中的极大强连通子图 称作有向图的**强连通分量**。

### 生成树和生成森林

生成树：一个含 n 个顶点的**连通图**的生成树是**该图中的一个极小连通子图**， 它包含图中 n 个顶点和足以构成一棵树的 n-1 条边。

对于**非连通图**，对其每个**连通分量**可以构造一棵生成树，**合成起来 就是一个生成森林**。

### 无向网和有向网

在实际应用中，图的弧或边往往与具有一定意义的数相关，称这些数 为“权” ，带权的图通常成为“网” ，分别称带权的有向图和无向图为 有向网和无向网。

## 7.2 图的存储结构 

### 邻接矩阵和加权邻接矩阵

![image-20241218232134000](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241218232134000.png)

![image-20241218232224192](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241218232224192.png)

```
// 图的数组(邻接矩阵)存储表示
#define INFINITY INT_MAX
#define MAX_VERTEX_NUM 20
typedef enum {DG, DN, UDG, UDN} GraphKind; // 类型标志{有向图,有向
网,无向图,无向网}

typedef struct ArcCell{ 
// 弧的定义
    VRType adj; 
    // VRType是顶点关系类型。
    // 对无权图，用1或0表示相邻否；对带权图，则为权值类型。
    InfoType *info; // 该弧相关信息的指针
} ArcCell, AdjMatrix[MAX_VERTEX_NUM][MAX_VERTEX_NUM];

typedef struct { 
    // 图的定义
    VertexType vexs[MAX_VERTEX_NUM]; // 顶点信息
    AdjMatrix arcs; // 表示顶点之间关系的二维数组
    int vexnum, arcnum; // 图的当前顶点数和弧(边)数
    GraphKind kind; // 图的种类标志
} MGraph;
```

顶点的“第一个”邻接点就应该是该顶点所 对应的行中值为非零元素的最小列号，其 “下一个”邻接点就是同行中离它最近的值 为非零元素的列号。

### 邻接表(*)

类似于树的孩子链表，将和同一顶点“相邻接”的所有邻接点 链接在一个单链表中，单链表的头指针则和顶点信息一起存储在一 个一维数组中。

![image-20241218232910683](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241218232910683.png)

```
#define MAX_VERTEX_NUM 20
typedef struct ArcNode{
	int adjvex;
    struct ArcNode *nextarc;
    VRType weight; // 与弧相关的权值，无权则为0
    InfoType *info; // 指向该弧相关信息的指针
}ArcNode;

typedef struct VNode{
    VertexType data; // 顶点信息
    ArcNode *firstarc; // 指向第一条依附该顶点的弧
}VNode, AdjList[MAX_VERTEX_NUM];

typedef struct {
    AdjList vertices; // 顶点数组
    int vexnum, arcnum; // 图的当前顶点数和弧数
    GraphKind kind; // 图的种类标志
}ALGraph;
```

### 十字链表

十字链表是有向图的另一种链式存储结构，目的是将在有向图 的邻接表和逆邻接表中两次出现的同一条弧用一个结点表示；

![image-20241218234020531](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241218234020531.png)

```
  #define MAX_VERTEX_NUM 20
  typedef struct ArcBox{
	//该弧的尾和头顶点的位置
  	int tailvex,headvex;
  	struct *ArcBox *hlink,*tlink;
  	VRType weight;
  	infoType *info;
  }ArcBox;
  
  typedef struct VexNode{
  	VertexType data;
  	ArcBox *firstin,*firstout;
  }VexNode;
  
  typedef struct{
  	VexNode xlist[MAX_VERTEX_NUM];
  	int vexnum,arcnum;
  	GraphKind kind;
  }OLGraph;
```

```
Status CreateDG(OLGraph &G){
	scanf(&G.vexnum,&G.arcnum,&IncInfo);
	for(i=0;i<G.vexnum;++i){
		scanf(&G.xlist[i].data);
		G.xlist[i].firstin = NULL;
		G.xlist[i].firstout = NULL;
	}
	for(k=0;k<G.arcnum;++k){
		scanf(&v1,&);
		i=LocateVex(G,v1);
		j=LocateVex(G,v2);
		p=(ArcBox*)malloc(sizeof(ArcBox));
		*p={
			i,j,
			G.xlist[j].firstin, 
			G.xlist[i].firstout, NULL
		}
		G.xlist[j].firstin 
		=G.xlist[i].firstout=p;
		if(IncInfo) Input (*p->info);
	}
}
```

**十字链表 = 邻接表 + 逆邻接表**

### 邻接多重表

![image-20241220155114938](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241220155114938.png)

```
    #define MAX_VERTEX_NUM 20
    typedef emnu {unvisited, visited} VisitIf;
    
    typedef struct Ebox {
        int ivex, jvex; 
        struct EBox *ilink, *jlink;
        VRType weight;
        InfoType *info; 
    } EBox;
    
    typedef struct VexBox { 
        VertexType data;
        Ebox *firstedge; 
    } VexBox;
    
    typedef struct {
        VexBox adjmulist[MAX_VERTEX_NUM];
        int vexnum, edgenum;
        GraphKind kind; 
    } AMLGraph;
```

## 7.3 图的遍历 

图的“遍历”是对图中的每个顶点都 进行一次访问且仅进行一次访问。

- (1)如何确保每个顶点都被访问到；
- (2)如何确保每个顶点只被访问一次。

### 一、深度优先搜索遍历图

（1）从图中某个顶点v出发，访问此顶点；

（2）从v未被访问的邻接点出发，深度优先遍历图，直至所有与v路径相同的顶点都 被访问；

（3）若仍有顶点尚未被访问，则另选图中一个未被访问的顶点出发，重复上述过程， 直至图中所有顶点都被访问为止。

![image-20241220155857179](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241220155857179.png)

```
  Boolean visited[MAX]; // 访问标志数组
  Status (* VisitFunc) (int v);
  void DFSTraverse(Graph G,Status (*visit)(int v)){
      // 对G作深度优先遍历
      VisitFunk=Visit;
      // 访问标识数组初始化
      for(v=0;v<G,vexnum;++v){
        visited[v]=False;
      }
      for(v=0;v<G,vexnum;++v){
		if(!visited[v]){
			//对尚未访问的顶点调用DFS
			DFS(G,v);
		}
      }
  }
  void DFS(Graph G,int v){
  	visited[v] = TRUE; VisitFunc(v);

for(w=FirstAdjVex(G,v);w>=0;w=NextAdjVex(G,v,w)){
  		if (!visited[w]) DFS(G,w);
  	}
  }
```

### 二、广度优先搜索遍历图

广度优先搜索(Breadth_First Search)的基本思想是：

- 从图中某个顶点 v 出发
- 在访问了 v 之后依次访问 v 的各个未曾访问过的邻接点
- 然后分别从这些邻接点出发依次访问它们的邻接点
- 并使得“先被访问的顶点的邻接点”先于“后被访问的顶点的邻接点” 进行访问，直至图中所有已被访问的顶点的邻接点都被访问到。
-  如若此时图中尚有顶点未被访问，则需另选一个未曾被访问过的顶点 作为新的起始点，重复上述过程，直至图中所有顶点都被访问到为止。

```
  void BFSTraverse(Graph G,Status(*visit)(int v)){
	for(v=0;v<G.vexnum;++v){
		initQueue(Q);
	}
	for(v=0;v<G.vexnum;++v){
		if (!visited[v]) { 
		// v尚未访问
			visited[v] = TRUE; 
			Visit(v);
			EnQueue(Q,v);
			while(!QueueEmpty(Q)){
				DeQueue(Q,u);
				for(w=FirstAdjVex(G,u);W>=0;w=NextAdjVex(G,u,w)){
				if (!visited[w]){
					visited[w]=TRUE;
					Visit(W);			
					EnQueue(Q,w);
					}
				}
			}
	}
  }
```

## 7.4 图的连通性问题 （一）

- 第一，对于连通图，一次遍历，便可访问图中所有顶点；对于非连通图，需要从多个顶点出发，一次遍历所能访问的顶点集 是对应连通分量顶点集。

- 第二，任何顶点仅被访问一次，也就是说 到任意顶点的路径仅有一条。即，对图进 行遍历的过程中，将边分成两个集合 T(E) 和 B(E)。其中 T(E) 中的边具有这样的特 性：通过它找到 “未被访问” 的邻接点。 显而易见，遍历访问所经路径T(E)与一次 遍历所能访问顶点集构成极小连通子图。对于连通图，该子图就是一棵生成树；对于非连通图，该子图是一个连通分量的 生成树，多次遍历得到生成森林。

```
    void DFSForest(Graph G, CSTree &T){
    
    }
```

```
    void DFSTree(Graph G, int v, CSTree &T){
    
    }
```

## 7.4 图的连通性问题 （二）

### 最小生成树

**最小生成树(Minimum Cost Spanning Tree)**：生成树中边的权 值（代价）之和最小的树。

MST 性质：假设 G = {V, { E } } 是一个连通图，U 是结点集合 V 的一 个非空子集。若 ( u, v ) 是一条代价最小的边，且 u 属于 U , v 属于 V-U，则必存在一棵包括边 ( u, v ) 在内的最小代 价生成树。

### 普里姆（Prim）算法

普里姆(Prim)算法和克鲁斯卡尔(Kruskal)算法是两个利用MST 性质构造最小生成树的算法。

**基本思想：**

- 将图中顶点集合V分成生成树顶点集U，和尚未落在生成树上的顶点集为 V-U
- 首先选取图中任意一个顶点 v 作为生成树的根
- 往生成树中添加顶点 w。 
- w 和v 之间必须有边，且权值在和 v 相邻接的边 中最小

顶点 w 应满足下列条件：它和生成树上的顶点之间的边上的权值是在连接这两 类顶点的所有边中权值属最小。

![image-20241220163841201](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241220163841201.png)

![image-20241220164053494](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241220164053494.png)

```
	
```

### 克鲁斯卡尔（ Kruskal ）算法

基本思想：为使生成树上总的权值之和达到最小，则应使每一 条边上的权值尽可能地小，自然应从权值最小的边选起，直至 选出 n-1 条互不构成回路的权值最小边为止。

- 首先构造一个只含 n 个顶点的森林
- 然后依权值从小到大从连通网中**选择不使森林中产生回路的边**加入到森林中去
- 直至该森林变成一棵树为止，这棵树便是连通网的最小生成树
- 生成树上不允许有回路，因此并非每一条居当前权值最小的边都可选

## 7.5 最短路径（*）

### 最短路径

#### 1、单源点路径问题

路径中 的第一个顶点为“源点” ，路径中的最后一个顶点为“终点”。

单源点路径问题是指，已知一个有向网和网中某个源点，求得从该源点到图中其它各个顶点之间的最短路径。

##### 迪杰斯特拉算法

![image-20241220170853590](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241220170853590.png)



# 第八章 查找

## 8.1 静态查找表

```
    ADT StaticSearchTable {
    
    CreateTable(&ST, n);
    操作结果：构造一个含 n 个数据元素的静态查找表 ST。
    
    DestroyTable(&ST);
    初始条件：静态查找表 ST 存在；
    操作结果：销毁查找表 ST。
    
    Search(ST, key);
    初始条件：静态查找表 ST 存在，key为和查找表中元素的关键字类型相同的给定值；
    操作结果：若ST中存在其关键字等于key 的数据元素，则函数值为该元素的值或在表中的位置，否则为“空”。
    
    Traverse(ST, Visit());
    初始条件：静态查找表 ST 存在，Visit 是对元素操作的应用函数；
    操作结果：按某种次序对ST的每个元素调用函数 visit() 一次且仅一次，一旦 visit() 失败，则操作失败。
    } ADT StaticSearchTable 
```

### 顺序表的查找

以顺序表或线性链表表示静态查找表，则Search函数可用顺 序查找来实现。本节中只讨论它在顺序存储结构模块中的实现。

```
	typedef struct{
	  // 数据元素存储空间基址
	  // 建表时按实际长度分配，0号单元留空
		ElemType *elem;
		int length;
	}
```

![image-20241220172849835](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241220172849835.png)

```
    int Search_Seq (SSTable ST, KeyType key){
    // 比较关键字是否相等；
    // 比较是否查找了所有元素；
    // 数据元素在顺序表ST中从位置1开始存储
    // 若找到，则函数值为该元素在表中的位置，否则为0
    // 设置“哨兵”:每次循环进行一次比较
    ST.length[0].key = key;
    for(i=ST.length;!EQ(ST.elem[i].key,key);--i){
    		return i;
    		// 找不到时，i 为0（“哨兵”的作用）
    		// 从后往前找
		}
    }
```

定义：查找过程中先后和给定值进行比较的关键字个数的期望值称作 查找算法的**平均查找长度**(Average Search Length)。

![image-20241220174250510](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241220174250510.png)

![image-20241220174304897](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241220174304897.png)

![image-20241220174502696](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241220174502696.png)

### 有序表的查找

折半查找(Binary Search)又称二分查找，其查找过程是，先确定待查记 录所在范围（区间），然后逐步缩小范围直至找到该记录，或者当查找 区间缩小到0也没有找到关键字等于给定值的记录为止。

在进行等概率(折半)查找时查找成功时的平均查找长度为：

![image-20241220175225555](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241220175225555.png)

### 索引顺序表的查找

“索引顺序查找”又称“分块查找” ，是顺序查找的一种改进方法；

建立“索引表” ，对每个子表建立“索引项” ；

“索引项”包括：

- **关键字项：其值为该子表内的最大关键字**
- **指针项：指示该子表的第一个记录在表中的位置**

![image-20241220175419773](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241220175419773.png)

查找过程：首先在索引表中进行折半或顺序查找，以确定待查 关键字在分块有序表中所在“块” ，块中记录是任意排列的， 在“块”中进行顺序查找。

在索引顺序表中进行索引顺序查找的平均查找长度为查找索引 表确定所在块的平均查找长度Lb和在块中查找元素的平均查找长度 Lw 之和。

![image-20241220175550680](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241220175550680.png)

![image-20241220175934017](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241220175934017.png)

## 8.2 动态查找表 

```
	ADT DynamicSearchTable {
    DestroyDSTable(&DT);
    初始条件：动态查找表DT存在；
    操作结果：销毁动态查找表 DT。
    
    SearchDSTable(DT, key);
    初始条件：动态查找表DT存在，key为和关键字类型相同的给定值；
    操作结果：若DT中存在其关键字等于key的数据元素，则函数值为该元素的值或在表中的位置，否则为“空”。
    
    InsertDSTable(&DT, e);
    初始条件：动态查找表DT存在，e 为待插入的数据元素；
    操作结果：若DT中不存在其关键字等于e.key的数据元素，则插入e 到DT。
    
    DeleteDSTable(&T, key);
    初始条件：动态查找表DT存在，key为和关键字类型相同的给定值；
    操作结果：若DT中存在其关键字等于key的数据元素，则删除之。
    
    TraverseDSTable(DT, Visit());
    初始条件：动态查找表DT存在，Visit 是对结点操作的应用函数；
    操作结果：按某种次序对DT的每个结点调用函数visit()一次且至多一次。一旦 visit() 失败，则操作失败。
    } ADT DynamicSearchTable
```

#### 二叉排序树

二叉排序树(Binary Sort Tree)，又称二叉查找树或者是一棵空树，或者是 具有如下特性的二叉树：

- 若它的左子树不空，则左子树上所有结点的值均小于根结点的值；
- 若它的右子树不空，则右子树上所有结点的值均大于根结点的值；

```
    BiTree SearchBST(BiTree T,KeyType key){
    	if(!T || EQ(key,T->data.key)) return(T);
    	else if LT(key,T->data.key){
    		return(SearchBST(T->lchild,key))
    	}else{
    		return(SearchBST(T->rchild,key))
    	}
    }
    
    
    Status SearchBST (BSTree T, KeyType key,BSTree f, BSTree &p){
    // 则指针p指向该数据元素结点,并返回TRUE,否则指针p指向查找路径上访
    // 问的最后一个结点并返回FALSE,指针f指向T的双亲,其初始调用值为NULL
		if(!T){
			p=f;
			return False;
		}else ifEQ(key,T->data.key){
			p = T; return TRUE;
		}else ifLT( key, T->data.key ){
			return SearchBST (T->lchild, key, T, p );
		}else{
			return SearchBST (T->rchild, key, T, p );
		}
    }
```

##### 二叉排序树的插入和删除

```
    Status InsertBST(BiTree &T, ElemType e ){
		// 当二叉查找树T中不存在关键字等于 e.key 的数据元素时，
		// 插入 e 并返回 TRUE，否则返回 FALSE
		if(!SearchBST ( T, e.key, NULL, p ){
		s = (BiTree)malloc(szeof(BiTNode));
        s->data = e; s->lchild = s->rchild = NULL;
            if ( !p ){
                T = s;
            }else if (LT(e.key,p->data.key)){
                p->lchild = s;
            }else{
                p->rchild = s;
            }
		return TRUE;
		}
		return FALSE;
    }
```



```
    Status DeleteBST (BiTree &T, KeyType key ) {
    // 若二叉查找树T中存在关键字等于key的数据元素时，则删除该数据元素结点 *p，并返回 TRUE；否则返回 FALSE
        if (!T) return FALSE; // 不存在关键字等于key的数据元素
        else {
        if (EQ( key, T->data.key) ) { return Delete (T); }
        // 找到关键字等于key的数据元素
        else if (LT( key, T->data.key)) return DeleteBST ( T->lchild, key );
        else return DeleteBST ( T->rchild, key );
        } // else
    } 
```

##### 二叉排序树的查找分析

## 8.3 哈希表

在记录的存储位置和它的关键字之间建立一个确定的对应关系f， 使每个关键字和结构中一个惟一的存储位置相对应。因而在查 找时，只要根据这个对应关系f找到给定值K的像f(K)。若结构中 存在关键字和K相等的记录，则必定在f(K)的存储位置上。

对应关系f为**哈希(Hash)函数**

表为**哈希表**

![image-20241220203627294](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241220203627294.png)

#### 一、哈希函数的构造方法

若对于关键字集合中的任意一个关键字，经哈希函数映象到地址 集合中任何一个地址的概率相等，则称此类哈希函数为**均匀的 (Uniform)哈希函数**。

##### 1、直接地址法

取关键字本身或关键字的某个线性函数值作为哈希表的地址。 即 Hash(key)=key 或 Hash(key)=a×key+b

##### 2、数字分析法

如果可能出现的关键字的数位相同，且取值事先知道，则可对关键字进行 分析

##### 3、平方取中法

如果关键字的所有各位分布都不均匀，则可取关键字的平方值的中间若干 位作为哈希表的地址。由于一个数的平方值的中间几位数受该数所有位影响， 将使随机分布的关键字得到的哈希函数值也随机。

##### 4、折叠法

若关键字的位数很多，且每一位上数字分布大致均匀，则可采用移位叠加 或间界叠加，即将关键字分成若干部分，然后以它们的叠加和(舍去进位) 作为哈希地址。移位叠加是将分割后的每一部分的最低位对齐，然后相加； 间界叠加是从一端向另一端沿分割界来回折叠，然后对齐相加。

![image-20241220204615196](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241220204615196.png)

##### 5、除留余数法

取关键字被某个不大于哈希表长m的数p除后所得余数作为哈希地址。即 Hash(key) = key MOD p，p≤m 

其中，MOD表示“取模”运算，p 为不大于表长的素数或不包含小于 20的质因数的合数。

##### 6、随机数法

当关键字不等长时，可取关键字的某个伪随机函数值作为哈希地址。 Hash(key) = random(key)

#### 二、处理冲突的方法

开放定址法：设法为发生冲突的关键字“找到”哈希表中 另一个尚未被记录占用的位置。

![image-20241220204824725](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241220204824725.png)

- 线性探测再散列：di= 1,2,3,…,m-1
- 二次探测再散列：
- ![image-20241220205205485](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241220205205485.png)
- 伪随机探测再散列：
- ![image-20241220205301283](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241220205301283.png)



![image-20241220205415781](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241220205415781.png)

# 第九章 内部排序

## 9.1 概述

## 9.2 插入排序

### 直接插入排序

![image-20241220210205300](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241220210205300.png)

设置 R[0] 为监视哨，并在查找的同时进行“记录后移” 

```
	void InsertSort(SqList &L){
		for ( i=2; i<=L.length; ++i ){
			if (L.r[i].key<L.r[i-1].key ){
				L.r[0] = L.r[i];
				L.r[i]=L.r[i-1];
				for ( j=i-2; L.r[0].key < L.r[j].key; --j ){
				L.r[j+1] = L.r[j]; }
				}
                L.r[j+1] = L.r[0];
			}
		}
	}
```

![image-20241220211228356](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241220211228356.png)

1.大于：不需要移动，直接插入； 

2.等于：不需要移动，直接插入； 

3.小于：当前元素向后搬移；

![image-20241220211356097](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241220211356097.png)

### 折半插入排序

### 2路插入排序

2-路插入排序是在折半插入排序的基础上再改进之，其目的是减少排序过 程中移动记录的次数，但为此需要n个记录的辅助空间。

![image-20241220212141495](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241220212141495.png)

### 表插入排序

假设有一个长度为 nnn 的数组 `A`，表插入排序的关键是构建一个指针数组 `link`，其中：

- `link[i]`：指向排序后的下一个元素的索引。
- 通过 `link` 的结构形成一个有序链表，最终遍历该链表即可输出排序后的结果。

### 希尔排序

**希尔排序**是一种基于插入排序的改进算法，它通过将整个待排序序列分为若干子序列进行排序，然后逐步缩小子序列的间隔（gap），最终在较小间隔下完成排序，优化了插入排序的性能。

<img src="C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241220212928254.png" alt="image-20241220212928254" style="zoom:80%;" />

- 初始化 gapgapgap 为数组长度的一半。
- 按 gapgapgap 将数组分为若干子序列，对每个子序列执行插入排序。
- 减小 gapgapgap 的值（通常为 ( gap = \lfloor gap / 2 \rfloor \））。
- 当 gap=1gap = 1gap=1 时，进行最后一次插入排序，结束算法。

## 9.3 快速排序

### 起泡排序

### 快速排序

快速排序是一种**分治法**（Divide and Conquer）的排序算法，通过递归地划分数组，将大问题拆解为小问题来完成排序。它通常被认为是高效的通用排序算法之一。

![image-20241220213608924](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241220213608924.png)

![image-20241220213631604](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241220213631604.png)

```
    int Partition(int arr[], int low, int high){
		int pivot=arr[high];
		int i=low-1;
		
		for(int j=low;j<high;j++){
			if(arr[j]<pivot){
				i++;
				int temp=arr[i];
				arr[i]=arr[j];
				arr[j]=temp;
			}
		}
		int temp = arr[i + 1];
        arr[i + 1] = arr[high];
        arr[high] = temp;
    }

    void QuickSort(int arr[], int low, int high) {
        if (low < high) {
            int pi = Partition(arr, low, high);
            QuickSort(arr, low, pi - 1); 
            QuickSort(arr, pi + 1, high); 
        }
    }
```

## 9.4 选择排序

### 简单选择排序

 通过反复从未排序部分中找到最小（或最大）的元素，将其放到已排序部分的末尾或开头，直至数组完全有序。

![image-20241220214352156](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241220214352156.png)

### 树形选择排序 

树形选择排序（Tree Selection Sort）是一种基于树形结构改进的选择排序算法。它通过构建一棵二叉树来快速找到最小值。

![image-20241220214453387](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241220214453387.png)

### 堆排序

堆排序是一种**选择排序**方法，其核心思想是通过**堆结构**快速选出无序序列中的最大值或最小值，并将其放到有序序列中。堆排序的关键在于**利用堆的特性**，减少每次选择过程中的比较次数和元素移动次数。

**堆的定义**：

- **大顶堆**：每个节点的值都大于或等于其子节点的值，堆顶为最大值。
- **小顶堆**：每个节点的值都小于或等于其子节点的值，堆顶为最小值。

**堆排序的思想**：

- 首先将无序序列构造成一个堆（大顶堆用于升序排序，小顶堆用于降序排序）。
- 通过多次将堆顶元素（最大或最小值）移出堆，并调整剩余元素使其重新满足堆的性质，最终得到一个有序序列。

![image-20241220215259285](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241220215259285.png)

#### **堆排序的关键问题**

1. **如何建堆？**
   - 从最后一个非叶子节点开始，利用“筛选”操作将每个子树调整为堆。
2. **如何调整堆？**
   - 每次将堆顶元素移除，用堆末尾元素替代，并从堆顶开始向下筛选，调整为大顶堆。

```
    // 堆调整函数：调整堆以满足大顶堆性质
    void HeapAdjust(int arr[], int s, int m){
		int rc=arr[s];
		for(int j=2*s+1;j<=m;j=2*j+1){
			if (j < m && arr[j] < arr[j + 1]) { 	// 如果右子节点更大，指向右子节点
            j++;
        	}
        	if (rc >= arr[j]) { // 如果根节点大于等于子节点，调整结束
            break;
        }
            arr[s] = arr[j]; 
            s = j;
		}
		arr[s]=rc;
    }
    
	// 堆排序函数
    void HeapSort(int arr[], int length){
		for (int i = length / 2 - 1; i >= 0; i--) {
		HeapAdjust(arr, i, length - 1);
		}
		
		for (int i = length - 1; i > 0; i--) {
		swap(arr[0], arr[i]);
		HeapAdjust(arr, 0, i - 1);
		}
    }
```

## 9.5 归并排序

归并排序是一种基于**分治法**的排序算法，核心思想是将数组分成多个小的子序列，每个子序列排序后合并，最终形成有序数组。归并排序是**稳定排序**，排序后的相同元素保持原来的相对位置。

![image-20241220222325839](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241220222325839.png)

```

    // 合并两个有序子序列
    void Merge(int arr[], int temp[], int left, int mid, int right) {
        int i = left;      // 左子序列的起始索引
        int j = mid + 1;   // 右子序列的起始索引
        int k = left;      // 合并后的索引

        // 合并两个子序列到临时数组
        while (i <= mid && j <= right) {
            if (arr[i] <= arr[j]) {
                temp[k++] = arr[i++];
            } else {
                temp[k++] = arr[j++];
            }
        }

        // 如果左子序列还有剩余元素，直接复制
        while (i <= mid) {
            temp[k++] = arr[i++];
        }

        // 如果右子序列还有剩余元素，直接复制
        while (j <= right) {
            temp[k++] = arr[j++];
        }

        // 将临时数组复制回原数组
        for (int m = left; m <= right; m++) {
            arr[m] = temp[m];
        }
    }

    // 递归归并排序
    void MergeSortRecursive(int arr[], int temp[], int left, int right) {
        if (left < right) {
            int mid = (left + right) / 2;

            // 对左子序列递归排序
            MergeSortRecursive(arr, temp, left, mid);

            // 对右子序列递归排序
            MergeSortRecursive(arr, temp, mid + 1, right);

            // 合并两个有序子序列
            Merge(arr, temp, left, mid, right);
        }
    }

    // 归并排序主函数
    void MergeSort(int arr[], int n) {
        int* temp = new int[n]; // 临时数组
        MergeSortRecursive(arr, temp, 0, n - 1);
        delete[] temp;
    }
```



## 9.6 基数排序

基数排序是一种**非比较型排序算法**，通过逐位处理关键字（从最低位到最高位或相反）来实现排序。其核心思想是将排序任务分解为多个小范围的子任务，分别对每一位进行“分配”和“收集”操作，最终完成整个序列的排序。

### 多关键字排序

每个记录被视为由多个关键字构成，关键字可以是数字的位数或字符串的字符。

按照“位”的顺序进行排序，可以有两种策略：

- **最高位优先（MSD）**：先按高位关键字排序，再对低位关键字排序。
- **最低位优先（LSD）**：先按低位关键字排序，再对高位关键字排序。

### 链式基数排序

## 9.7 各种内部排序方法的比较讨论

空间性能

排序方法的稳定性能

![image-20241220223144672](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241220223144672.png)

![image-20241220223206877](C:/Users/Y9000p/AppData/Roaming/Typora/typora-user-images/image-20241220223206877.png)