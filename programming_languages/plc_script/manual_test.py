'''
This file is used for debugging
'''

from parsing.ll1 import LL1
from interpretation.ast_interpreter import ASTInterpreter

plc_script = '''
    a = 2;    
    
    printIt("Starting value of a:");
    printIt(a);
    
    while (a < 5) {
        a = a + 1;
        printIt(a);        
    }
    
    if (a<5) {
        printIt("a is less than 5");
    }
    
    if (a == 5) {
        printIt("a is exactly 5!");
    }
    
    if (a > 6) {
        printIt("a is greater than 5!");
    }
    
    printIt("Final value of a:");
    printIt(a);
    
'''

ll1 = LL1(plc_script)
ast = ll1.parse()

print(ast)
interpreter = ASTInterpreter()
ret = interpreter.interpret(ast)