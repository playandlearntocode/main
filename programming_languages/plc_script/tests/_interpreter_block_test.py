from interpretation.ast_interpreter import ASTInterpreter


def test():
    ast = {
        'type': 'block',

        'content': {
            'type': 'number',
            'value': '2'
        }
    }

    interpreter = ASTInterpreter()
    ret = interpreter.interpret(ast)
