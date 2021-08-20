from parsing.ll1 import LL1
from interpretation.ast_interpreter import ASTInterpreter


def test():
    s = '''
    someVar = 10;
    a = 2;
    b = 3;
    
    function f1(a,b){
        return a*b;
    }
    
    c = someVar + f1(a,b);

    '''

    ll1 = LL1(s)
    ast = ll1.parse()


    interpreter = ASTInterpreter()
    ret = interpreter.interpret(ast)

    assert (ret == 16)
