from parsing.ll1 import LL1


def test():
    s = '''
    
    2 + 3 * 4 +7;
    
    '''

    ll1 = LL1(s)
    ast = ll1.parse()

    expected = {
        'type': 'addition_expression',
        'content': {
            'left': {
                'type': 'addition_expression',
                'content': {
                    'left': {
                        'type': 'number',
                        'value': '2'
                    },
                    'right': {
                        'type': 'multiplication_expression',
                        'content': {
                            'left': {
                                'type': 'number',
                                'value': '3'
                            },
                            'right': {
                                'type': 'number',
                                'value': '4'
                            }
                        }
                    }
                }
            },
            'right': {
                'type': 'number',
                'value': '7'

            }
        }
    }

    print('ast:')
    print(ast)

    assert (ast == expected)
