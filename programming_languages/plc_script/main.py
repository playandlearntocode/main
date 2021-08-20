'''
Sample plc_script program
Author:
Goran Trlin
https://playandlearntocode.com
'''

from parsing.ll1 import LL1
from interpretation.ast_interpreter import ASTInterpreter

# small plc_script program to parse and interpret:
plc_script = '''
// variable declaration:
someVar = 10;
a = 2;
b = 3;

// user-defined function:
function multiply(a,b){
    return a*b;
}

function sum(a,b) {
    return a+b;
}

printIt("Heej ho!");
c = someVar + (multiply(a,b) + sum(a,b));
printIt(c);
'''

ll1 = LL1(plc_script)
ast = ll1.parse()

interpreter = ASTInterpreter()

print('Starting program interpretation')
ret = interpreter.interpret(ast)
print('Program interpreted!')