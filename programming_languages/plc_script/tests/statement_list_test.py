from parsing.ll1 import LL1


def test():
    s = '{2;3;4;}'

    ll1 = LL1(s)
    ast = ll1.parse()

    expected = {
        'type': 'block',
        'content': [
            {
                'type': 'number',
                'value': '2'
            },
            {
                'type': 'number',
                'value': '3'
            },
            {
                'type': 'number',
                'value': '4'
            }
        ]
    }

    assert (ast == expected)
