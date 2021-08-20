from interpretation.ast_interpreter import ASTInterpreter


def test():
    ast = [
        {
            'type': 'function_declaration_statement',
            'name': 'mix',
            'arguments': [
                {'type': 'identifier',
                 'value': 'a'
                 },
                {'type': 'identifier',
                 'value': 'b'
                 }
            ],
            'content': {
                'type': 'block',
                'content': {
                    'type': 'return_statement',
                    'content':
                        {
                            'type': 'addition_expression',
                            'content': {
                                'left': {
                                    'type': 'identifier', 'value': 'a'},
                                'right': {
                                    'type': 'identifier', 'value': 'b'}
                            }
                        }
                }

            }
        },
        {
            'type': 'function_call_expression',
            'callee': 'mix',
            'arguments': [
                {'type': 'number',
                 'value': '2'
                 },
                {'type': 'number',
                 'value': '3'
                 }
            ]
        }
    ]

    interpreter = ASTInterpreter()
    ret = interpreter.interpret(ast)

    assert(ret==5)