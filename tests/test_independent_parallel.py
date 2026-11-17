"""Independent reference geometry and annotation-coordinate checks."""
import unittest
import numpy as np
import nibabel as nib
import torch
from experiments.validate_independent_parallel import annotation_points, rotation, spatial_projection, require_qualification


class IndependentParallelTests(unittest.TestCase):
    def test_annotation_world_points_preserve_flips_and_axis_permutation(self):
        image=nib.Nifti1Image(np.zeros((3,4,5)),np.diag([2.,3.,4.,1.]))
        annotations=[{'direction':['S','P','L']},{'label':1,'X':1.,'Y':2.,'Z':.5}]
        # S is original z, P reverses y, L reverses x.
        np.testing.assert_allclose(annotation_points(image,annotations),[[3.,3.,4.]])

    def test_independent_rotation_and_spatial_line_integrals(self):
        torch.testing.assert_close(rotation(torch.tensor([90.,0.,0.])),
            torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]),atol=1e-6,rtol=0)
        volume=torch.ones(5,5,5)
        for view in ['ap','lateral']:
            image=spatial_projection(volume,(1.,1.,1.),torch.zeros(6),view,5,1.,5.,ray_step=.25)
            torch.testing.assert_close(image,torch.full((1,1,5,5),5.),atol=.02,rtol=0)

    def test_failed_cuda_qualification_cannot_be_complete(self):
        with self.assertRaisesRegex(ValueError,'qualification'):
            require_qualification([{'dtype':'torch.float32','passed':False}])

    def test_invalid_and_duplicate_annotations_are_rejected(self):
        image=nib.Nifti1Image(np.zeros((3,4,5)),np.eye(4))
        for annotations in [
            [{'direction':['R','A','S']},{'label':1,'X':10,'Y':0,'Z':0}],
            [{'direction':['R','A','S']},{'label':1,'X':0,'Y':0,'Z':0},{'label':1,'X':1,'Y':1,'Z':1}]]:
            with self.assertRaises(ValueError):annotation_points(image,annotations)

    def test_asymmetric_spatial_reference_anisotropy_and_physical_shifts(self):
        volume=torch.zeros(7,9,11);volume[4,2,7]=1.;volume[2,5,4]=.4
        spacing=(3.,2.,1.);res=41;pitch=1.
        # At 90deg about z, scanner x=-object y, y=object x.
        pose=torch.tensor([90.,0.,0.,1.,-1.,2.])
        expected={'ap':[(25,21,3.),(19,18,1.2)],'lateral':[(15,21,2.),(21,18,.8)]}
        for view,peaks in expected.items():
            image=spatial_projection(volume,spacing,pose,view,res,pitch,25.,ray_step=.125)[0,0]
            for col,row,value in peaks:self.assertAlmostEqual(float(image[row,col]),value,places=3)

    def test_spatial_quadrature_against_oblique_analytic_gaussian(self):
        from scipy.spatial.transform import Rotation
        spacing=(1.1,.9,.7)
        z,y,x=torch.meshgrid(*(torch.arange(-20,21)*s for s in spacing),indexing='ij')
        mean=np.array([2.,-1.,3.]);sigma=np.array([2.5,3.6,2.8])
        volume=torch.exp(-.5*(((torch.stack([x,y,z],-1)-torch.tensor(mean))/torch.tensor(sigma))**2).sum(-1)).float()
        pose=torch.tensor([30.,17.,9.,.37,-.23,.45])
        independent=Rotation.from_euler('ZYX',[30,17,9],degrees=True).as_matrix()
        covariance=np.diag(sigma**2)
        vv,uu=np.meshgrid(np.arange(-25,26),np.arange(-25,26),indexing='ij');uv=np.stack([uu,vv],-1)
        for view,basis in [('ap',np.array([[1,0,0],[0,1,0]])),('lateral',np.array([[0,0,-1],[0,1,0]]))]:
            cov=basis@independent@covariance@independent.T@basis.T
            center=basis@(independent@mean+pose[3:].numpy())
            delta=uv-center
            expected=np.sqrt(2*np.pi*np.linalg.det(covariance)/np.linalg.det(cov))*np.exp(-.5*np.einsum('...i,ij,...j->...',delta,np.linalg.inv(cov),delta))
            actual=spatial_projection(volume,spacing,pose,view,51,1.,30.,ray_step=.175)[0,0].numpy()
            self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),.04)


if __name__=='__main__':unittest.main()
