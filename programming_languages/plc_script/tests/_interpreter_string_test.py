from interpretation.ast_interpreter import ASTInterpreter


def test():
    ast = {
        'type': 'string',
        'value': '2'
    }

    interpreter = ASTInterpreter()
    ret = interpreter.interpret(ast)
