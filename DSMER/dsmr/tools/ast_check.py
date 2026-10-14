import ast
import sys
path = sys.argv[1]
try:
    src = open(path, 'r', encoding='utf-8').read()
    ast.parse(src, filename=path)
    print('PARSE_OK')
except SyntaxError as e:
    print('SYNTAX_ERROR')
    print(e)
    raise
except Exception as e:
    print('ERROR')
    print(e)
    raise
