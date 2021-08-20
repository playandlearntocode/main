from interpretation.ast_interpreter import ASTInterpreter


def test():
    ast = {
        'type': 'addition_expression',
        'content': {
            'left': {
                'type': 'addition_expression',
                'content': {
                    'left': {
                        'type': 'number', 'value': '2'},
                    'right': {
                        'type': 'number', 'value': '3'}
                }
            },
            'right': {
                'type': 'number', 'value': '4'}
        }
    }

    interpreter = ASTInterpreter()
    ret = interpreter.interpret(ast)

    assert(ret == 9)
