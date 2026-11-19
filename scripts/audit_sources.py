"""Static implemented-entrypoint audit; standard library only, no training imports."""
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def audit(root=ROOT):
    files=sorted(list((root/'ops_repro').glob('*.py'))+list((root/'src').glob('*.py'))+list((root/'scripts').glob('*.py')))
    findings=[];functions=0
    for path in files:
        tree=ast.parse(path.read_text(),filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                functions+=1
                body=[n for n in node.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))]
                if not body or all(isinstance(n,ast.Pass) or isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and n.value.value is Ellipsis for n in body):
                    findings.append(dict(file=str(path.relative_to(root)),line=node.lineno,issue='empty function'))
            if isinstance(node,ast.Constant) and node.value is Ellipsis:findings.append(dict(file=str(path.relative_to(root)),line=node.lineno,issue='ellipsis'))
            if isinstance(node,ast.Name) and node.id=='NotImplementedError':findings.append(dict(file=str(path.relative_to(root)),line=node.lineno,issue='not implemented'))
            if isinstance(node,ast.Pass):findings.append(dict(file=str(path.relative_to(root)),line=node.lineno,issue='pass statement'))
    return dict(files=len(files),functions=functions,findings=findings,scope='Active code only; this AST audit complements, and cannot replace, behavior tests and platform GPU qualification')

if __name__=='__main__':
    result=audit();print(json.dumps(result,ensure_ascii=False,indent=2));raise SystemExit(bool(result['findings']))
