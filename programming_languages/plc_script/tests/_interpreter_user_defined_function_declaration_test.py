from interpretation.ast_interpreter import ASTInterpreter


def test():
    ast = {
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
    }

    interpreter = ASTInterpreter()
    ret = interpreter.interpret(ast)
