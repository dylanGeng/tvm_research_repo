//===----------------------------------------===//
//	Abstract Syntax Tree (aka Parse Tree)
//===----------------------------------------===//

/// ExprAST - Base class for all expression nodes
class ExprAST {
public:
	virtual ~ExprAST() {}
};

/// NumberExprAST - Expression class for numeric literals like "1.0"
class NumberExprAST : public ExprAST
{
	double Val;
public:
	NumberExprAST(double val): Val(val) {}
	~NumberExprAST();
	
};

