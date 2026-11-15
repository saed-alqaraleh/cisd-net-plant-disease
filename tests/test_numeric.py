"""Dependency-light regression tests for the released numeric primitives."""
import sys
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cisd_net import JointStiefelCayleyTransform, compute_aurcc

class NumericTests(unittest.TestCase):
    def test_joint_orthogonal_frame(self):
        t=JointStiefelCayleyTransform(64,8,4,seed=42)
        _,up,ub=t.forward()
        self.assertLess(np.linalg.norm(up.T@ub),1e-12)
        self.assertLess(np.linalg.norm(np.column_stack([up,ub]).T@
                                       np.column_stack([up,ub])-np.eye(12)),1e-12)

    def test_aurcc_perfect(self):
        probs=np.array([[0.9,0.1],[0.2,0.8]],dtype=float)
        self.assertEqual(compute_aurcc(probs,np.array([0,1])),0.0)

if __name__=='__main__': unittest.main()
