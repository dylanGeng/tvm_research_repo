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

/// gettok - Return the next token from standard input.
static int gettok() {
	static int LastChar = ' ';

	// Skip any whitespace.
	while(isspace(LastChar))
		LastChar = getchar();

	if(isalpha(LastChar)) {	// identifier: [a-zA-Z][a-zA-Z0-9]*
		IdentifierStr = LastChar;
		
		while(isalnum(LastChar = getChar())){
			IdentifierStr += LastChar;
		}

		if(IdentifierStr == "def")
			return tok_def;
		if(IdentifierStr == "extern")
			return tok_extern;
		return tok_identifier;
	}

	if(isdigit(LastChar) || LastChar == '.'){ // Number: [0-9.]+
		std::string NumStr;
		do {
			NumStr += LastChar;
			LastChar = getChar();
		} while (isdigit(LastChar) || LastChar == '.');

		NumVal = strtod(NumStr.c_str(), nullptr);
		return tok_number;
	}

	if(LastChar == '#') {
		// Comment until end of line.
		do
			LastChar = getChar();
		while(LastChar != EOF && LastChar != '\n' && LastChar != '\r');

		if(LastChar != EOF)
			return gettok();
	}

	// Check for end of file. Don't eat the EOF
	if(LastChar != EOF)
		return tok_eof;

	// Otherwise, just return the character as its ascii value.
	int ThisChar = LastChar;
	LastChar = getChar();
	return ThisChar;
}

//===----------------------------------------===//
//	Abstract Syntax Tree (aka Parse Tree)
//===----------------------------------------===//
namespace {
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
} // end anonymous namespace

auto LHS = llvm::make_unique<VariableExprAST>("x");
auto RHS = llvm::make_unique<VariableExprAST>("y");
auto Result = std::make_unique<BinaryExprAST>('+', std::move(LHS),
 std::move(RHS));

//===--------------------------------------------------------------------------------===//
// Parser
//===--------------------------------------------------------------------------------===//

/// CurTok/getNextToken - Provide a simple token buffer. CurTok is the current
/// token the parset is looking at. getNextToken reads another token from the 
/// lexer and updates CurTok with its results.
static int CurTok;
static int getNextToken() {
	return CurTok = gettok();
}

/// BinopPrecedence - This holds the precedence for each binary operator that is 
/// defined.
static std::map<char, int> BinopPrecedence;

/// GetTokPrecedence - Get the precedence of the pending binary operator token.
static int GetTokPrecedence() {
	if (!isascii(CurTok))
		return -1;
	
	// Make sure it's a declared binop
	int TokPrec = BinopPrecedence[CurTok];
	if (TokPrec <= 0)
		return -1;
	return TokPrec;
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

/// identifierexpr
///   ::= identifier
///   ::= identifier '(' expression ')'
static std::unique_ptr<ExprAST> ParseIdentifierExpr() {
	std::string IdName = IdentifierStr;

	getNextToken(); //eat identifier.

	if (CurTok != '(') //Simple variable ref.
		return llvm::make_unique<VariableExprAST>(IdName);
	
	// Call.
	getNextToken();	//eat (
	std::vector<std::unique_ptr<ExprAST> > Args;
	if (CurTok != ')') {
		while (1) {
			if (auto Arg = ParseExpression())
				Args.push_back(std::move(Arg));
			else
				return nullptr;

			if (CurTok == ')')
				break;
			
			if (CurTok != ',')
				return LogError("Expected ')' or ',' in argument list");
			getNextToken();
		}
	}

	// Eat the ')'.
	getNextToken();

	return llvm::make_unique<CallExprAST>(IdName, std::move(Args));
}

/// primary
///		::= identifierexpr
///		::= numberexpr
///		::= parenexpr
static std::unique_ptr<ExprAST> ParsePrimary() {
	switch (CurTok) {
	default:
		return LogError("unknown token when expecting an expression");
	case tok_identifier:
		return ParseIdentifierExpr();
	case tok_number:
		return ParseNumberExpr();
	case '(':
		return ParseParenExpr();
	}
}

/// binoprhs
///		::= ('+' primary)*
static std::unique_ptr<ExprAST> ParseBinOpRHS(int ExprPrec, std::unique_ptr<ExprAST> LHS) {

	// If this is a binop, find its precedence.
	while (true) {
		int TokPrec = GetTokPrecedence();

		// If this is a binop that binds at least as tightly as the current binop,
		// consume it, otherwise we are done.
		if (TokPrec < ExprPrec)
			return LHS;
		
		// Okay, we know this is a binop.
		int BinOp = CurTok;
		getNextToken(); // eat binop

		// Parse the primary expression after the binary operator.
		auto RHS = ParsePrimary();
		if(!RHS)
			return nullptr;

		// If BinOp binds less tightly with RHS than the operator after RHS, let
		// the pending operator take RHS as its LHS.
		int NextPrec = GetTokPrecedence();
		if (TokPrec < NextPrec) {
			RHS = ParseBinOpRHS(TokPrec+1, std::move(RHS));
			if (!RHS)
				return nullptr;
		}

		// Merge LHS/RHS
		LHS = llvm::make_unique<BinaryExprAST>(BinOp, std::move(LHS), std::move(RHS));
	}
}

/// expression
///		::= primary binoprhs
///
static std::unique_ptr<ExprAST> ParseExpression() {
	auto LHS = ParsePrimary();
	if(!LHS)
		return nullptr;

	return ParseBinOpRHS(0, std::move(LHS));
}

/// prototype
///		::= id '(' id* ')'
static std::unique_ptr<PrototypeAST> ParsePrototype() {
	if (CurTok != tok_identifier)
		return LogErrorP("Expected function name in prototype");
	
	std::string FnName = IdentifierStr;
	getNextToken();

	if( CurTok != '(')
		return LogErrorP("Expected '(' in prototype");

	// Read the list of argument names.
	std::vector<std::string> ArgNames;
	while (getNextToken() == tok_identifier)
		ArgNames.push_back(IdentifierStr);
	if (CurTok != ')')
		return LogErrorP("Expected ')' in prototype");
	
	//success
	getNextToken();	//eat ')'.

	return llvm::make_unique<PrototypeAST>(FnName, std::move(ArgNames));
}

/// definition ::= 'def' prototype expression
static std::unique_ptr<FunctionAST> ParseDefinition() {
	getNextToken();	//eat def.
	auto Proto = ParsePrototype();
	if (!Proto)
		return nullptr;
	if (auto E = ParseExpression() )
		return llvm::make_unique<FunctionAST>(std::move(Proto), std::move(E));
	return nullptr;
}

/// external ::= 'extern' prototype
static std::unique_ptr<PrototypeAST> ParseExtern() {
	getNextToken();	//eat extern
	return ParsePrototype();
}

/// toplevelexpr ::= expression
static std::unique_ptr<FunctionAST> ParseTopLevelExpr() {
	if (auto E = ParseExpression()) {
		// Make an anonymous proto
		auto Proto = llvm::make_unique<PrototypeAST>("", std::vector<std::string>());
		return llvm::make_unique<FunctionAST>(std::move(Proto), std::move(E));
	}
	return nullptr;
}

/// top ::= definition | external | expression | ';'
static void MainLoop() {
	while(1) {
		fprintf(stderr, "ready> ");
		switch(CurTok) {
			case tok_eof:
				return;
			case ';':
				getNextToken();
				break;
			case tok_def:
				HandleExtern();
				break;
			default:
				HandleTopLevelExpression();
				break;
		}
	}
}


//===--------------------------------------===//
//	Main driver code.
//===--------------------------------------===//
int main() {
	// Install standard binary operators.
	// 1 is lowest precedence.
	BinopPrecedence['<'] = 10;
	BinopPrecedence['+'] = 20;
	BinopPrecedence['-'] = 20;
	BinopPrecedence['*'] = 40; //highest

	// Prime the fist token.
	fprintf(stderr, "ready> ");
	getNextToken();

	// Run the main "interpreter loop" now.
	MainLoop();
	
	return 0;
}