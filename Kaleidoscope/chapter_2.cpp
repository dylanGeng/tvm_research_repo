#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <vector>

//===----------------------------------------------------===//
//	Lexer
//===----------------------------------------------------===//
// The lexer return tokens [0-255] if it is an unkown character, otherwise one
// of these for known things.
enum Token {
	tok_eof = -1,

	// commands
	tok_def = -2,
	tok_extern = -3,

	// primary
	tok_identifier = -4,
	tok_number = -5
};

static std::string IdentifierStr; //Filled in if tok_identifier
static double NumVal;	//Filled in if tok_number

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

/// VariableExprAST - Expression class for referenceing a variable, like "a".
class VariableExprAST : public ExprAST {
	std::string Name;
public:
	VariableExprAST(const std::string &name): Name(name) {}
};

/// BinaryExprAST - Expression class for a binary operator.
class BinaryExprAST : public ExprAST {
	char Op;
	std::unique_ptr<ExprAST> LHS, RHS;
public:
	BinaryExprAST(char op, std::unique_ptr<ExprAST> lhs, std::unique_ptr<ExprAST> rhs)
		: Op(op), LHS(std::move(lhs), RHS(std::move(rhs) {}
};

// CallExprAST - Expression class for function calls
class CallExprAST : public ExprAST {
	std::string Callee;
	std::vector<std::unique_ptr<ExprAST> > Args;
public:
	CallExprAST(const std::string &callee, std::vector<std::unique_ptr<ExprAST> > &args)
		: Callee(callee), Args(std::move(args)) {}
};

/// Prototype AST - This class represents the "prototype" for a function.
/// which captures its name, and its argument names(thus implicitly the number
/// of arguments the function takes).
class PrototypeAST {
	std::string Name;
	std::vector<std::string> Args;
public:
	PrototypeAST(const std::string &name,
	 const std::vector<std::string> &args)
	 : Name(name), Args(std::move(args) {}
};

/// FunctionAST - This class represents a function definition itself.
class FunctionAST {
	std::unique_ptr<PrototypeAST> Proto;
	std::unique_ptr<ExprAST> Body;
public:
	FunctionAST(std::unique_ptr<PrototypeAST> proto, std::unique_ptr<ExprAST> body)
	 : Proto(std::move(proto)), Body(std::move(body)) {}
};

auto LHS = llvm::make_unique<VariableExprAST>("x");
auto RHS = llvm::make_unique<VariableExprAST>("y");
auto Result = std::make_unique<BinaryExprAST>('+', std::move(LHS),
 std::move(RHS));

/// CurTok/getNextToken - Provide a simple token buffer. CurTok is the current
/// token the parset is looking at. getNextToken reads another token from the 
/// lexer and updates CurTok with its results.
static int CurTok;
static int getNextToken() {
	return CurTok = gettok();
}

/// LogError* - These are little helper functions for error handling.
std::unique_ptr<ExprAST> LogError(const  char *Str) {
	fprintf(stderr, "LogError: %s\n", Str);
	return nullptr;
}
std::unique_ptr<PrototypeAST> LogErrorP(const char *Str) {
	LogError(Str);
	return nullptr;
}

/// numberexpr ::= number
static std::unique_ptr<ExprAST> ParseNumberExpr() {
	auto Result = llvm::make_unique<NumberExprAST>(NumVal);
	getNextToken();	//consume the number
	return std::move(Result);
}

/// parenexpr ::= '(' expression ')'
static std::unique_ptr<ExprAST> ParseParenExpr() {
	getNextToken(); // eat(.
	auto V = ParseExpression();
	if(!V)
		return nullptr;
	if (CurTok != ')')
		return LogError("expected ')'");
	getNextToken(); //eat ).
	return V;
}

/// primary
///		::= identifierexpr
///		::= numberexpr
///		::= parenexpr
static std::unique_ptr<ExprAST> ParsePrimary() {
	switch (CurTok) {
	default:
		return LogError("unknown token when expecting an expression");
	case tok_
	}
}