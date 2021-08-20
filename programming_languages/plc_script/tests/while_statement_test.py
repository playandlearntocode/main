from parsing.ll1 import LL1


def test():
    s = '''
    a = 2;
    
    while (a < 5) {
        a = a + 1;
    }
    
    '''

    ll1 = LL1(s)
    ast = ll1.parse()

    expected = [
        {
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
            'type': 'while_statement',
            'condition': {
                'type': 'operator_less_than',
                'content': {
                    'left': {
                        'type': 'identifier',
                        'value': 'a'
                    },
                    'right': {
                        'type': 'number',
                        'value': '5'
                    }
                }
            },
            'while_body': {
                'type': 'block',
                'content': {
                    'type': 'assignment_expression',
                    'content': {
                        'left': {
                            'type': 'identifier',
                            'value': 'a'
                        },
                        'right': {
                            'type': 'addition_expression',
                            'content': {
                                'left': {
                                    'type': 'identifier', 'value': 'a'},
                                'right': {
                                    'type': 'number', 'value': '1'}
                            }
                        }
                    }
                }
            }
        }
    ]

    assert (ast == expected)
