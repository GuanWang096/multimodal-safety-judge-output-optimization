"""CPU regression of the adopted identity and voting-retirement guards.

No sockets, Docker, GPU, production private keys or performance measurements.
"""
from pathlib import Path
import json
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src/rggpbft'
OUT = ROOT / 'results/reproduced/revision/adopted-checks'
sys.path.insert(0, str(SRC))
import protocol
from identity_checks import ingress, load
from retirement_checks import targeted

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    identity = OUT/'identity'
    retirement = OUT/'retirement'
    identity.mkdir()
    retirement.mkdir()
    ingress_records = ingress(SRC, protocol, 'candidate', identity)
    retirement_records = targeted(SRC, 'candidate', retirement)
    suite = unittest.TestSuite()
    for name in ('test_protocol','test_view_change','test_view_runtime'):
        module = load('revision_'+name, ROOT/'src/revision/tests'/(name+'.py'))
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    assert result.wasSuccessful() and result.testsRun == 30
    report = dict(identity_cases=len(ingress_records),
                  identity_checks=sum(len(x['checks']) for x in ingress_records),
                  retirement_cases=len(retirement_records),
                  retirement_checks=sum(len(x['checks']) for x in retirement_records),
                  regression_tests=result.testsRun,
                  lower_view_vote_cases=sum(x['observations']['lower_view_new_vote'] for x in retirement_records))
    assert report['retirement_cases']==34 and report['retirement_checks']==80
    assert report['lower_view_vote_cases']==0
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))

if __name__ == '__main__':
    main()
