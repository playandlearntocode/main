from interpretation.ast_interpreter import ASTInterpreter


def test():
    ast = [{
        'type': 'assignment_expression',
        'content': {
            'left': {
                'type': 'identifier',
                'value': 'a'
            },
            'right': {
                'type': 'number',
                'value': '2'
            }
        }
    },
        {
            'type': 'addition_expression',
            'content': {
                'left': {
                    'type': 'identifier', 'value': 'a'},
                'right': {
                    'type': 'number', 'value': '2'}
            }
        }
    ]

    interpreter = ASTInterpreter()
    ret = interpreter.interpret(ast)

    assert (ret == 4)
