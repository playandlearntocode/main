from interpretation.ast_interpreter import ASTInterpreter


def test():
    ast = {
        'type': 'subtraction_expression',
        'content': {
            'left': {
                'type': 'subtraction_expression',
                'content': {
                    'left': {
                        'type': 'number', 'value': '24'},
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

    assert(ret == 17)
