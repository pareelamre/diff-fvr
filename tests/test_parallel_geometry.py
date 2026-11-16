"""Independent spatial references, equivariance and numerical-gradient checks."""
import unittest

import torch
import torch.nn.functional as F

from src.fvr.torch_engine import TorchDiffFVREngine


class ParallelGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.volume=torch.zeros(9,9,9)
        cls.volume[2,3,5]=1.
        cls.volume[6,2,3]=2.
        cls.volume[4,6,6]=.5

    def engine(self,spacing=(1.,1.,1.)):
        return TorchDiffFVREngine(self.volume,spacing=spacing,pad_factor=2.,use_triton=False,detector_spacing=min(spacing))

    def test_canonical_axis_projections_equal_spatial_line_sums(self):
        engine=self.engine()
        # Preserve parity when padding so the central voxel remains the origin.
        padded=F.pad(self.volume,(5,5,5,5,5,5))
        zero=torch.zeros(1,6)
        ap=engine(zero,view='ap',out_res=19)[0,0]
        lateral=engine(zero,view='lateral',out_res=19)[0,0]
        torch.testing.assert_close(ap,padded.sum(0),atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(lateral,padded.sum(2).T.flip(-1),atol=2e-5,rtol=2e-5)

    def test_physical_translations_and_independent_view_null_axes(self):
        engine=self.engine()
        zero=torch.zeros(1,6)
        shifted=torch.tensor([[0.,0.,0.,2.,-1.,3.]])
        ap=engine(zero,view='ap',out_res=19)
        lateral=engine(zero,view='lateral',out_res=19)
        torch.testing.assert_close(engine(shifted,view='ap',out_res=19),torch.roll(ap,(-1,2),(-2,-1)),atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(engine(shifted,view='lateral',out_res=19),torch.roll(lateral,(-1,-3),(-2,-1)),atol=2e-5,rtol=2e-5)
        tz=torch.tensor([[0.,0.,0.,0.,0.,3.]])
        tx=torch.tensor([[0.,0.,0.,2.,0.,0.]])
        torch.testing.assert_close(engine(tz,view='ap',out_res=19),ap)
        torch.testing.assert_close(engine(tx,view='lateral',out_res=19),lateral)
        self.assertGreater(float((engine(tz,view='lateral',out_res=19)-lateral).abs().max()),.4)

    def test_known_rotation_matches_rotated_spatial_projection(self):
        engine=self.engine()
        ap=engine(torch.zeros(1,6),out_res=19)
        rotated=engine(torch.tensor([[90.,0.,0.,0.,0.,0.]]),out_res=19)
        torch.testing.assert_close(rotated,torch.rot90(ap,-1,(-2,-1)),atol=2e-5,rtol=2e-5)

    def test_anisotropic_beam_spacing_sets_line_integral_units(self):
        engine=self.engine((2.,1.,1.))
        expected=F.pad(self.volume,(5,5,5,5,5,5)).sum(0)*2.
        torch.testing.assert_close(engine(torch.zeros(1,6),out_res=19)[0,0],expected,atol=2e-5,rtol=2e-5)

    def test_uniform_change_of_length_units_preserves_projection_up_to_integral_scale(self):
        a=self.engine((1.7,1.3,.9));b=self.engine((3.4,2.6,1.8))
        pose=torch.tensor([[3.,-4.,2.,1.,-.5,2.]])
        scaled=pose.clone();scaled[:,3:]*=2
        for view in ['ap','lateral']:
            torch.testing.assert_close(a(pose,view=view,out_res=31),b(scaled,view=view,out_res=31)/2,atol=2e-5,rtol=2e-5)

    def test_even_grid_nyquist_reflection_preserves_axis_projections(self):
        volume=torch.zeros(8,8,8);volume[2,3,5]=1.;volume[6,2,3]=2.
        engine=TorchDiffFVREngine(volume,pad_factor=2.,use_triton=False,detector_spacing=1.)
        padded=F.pad(volume,(4,4,4,4,4,4))
        zero=torch.zeros(1,6)
        torch.testing.assert_close(engine(zero,out_res=16)[0,0],padded.sum(0),atol=2e-5,rtol=2e-5)
        expected=torch.roll(padded.sum(2).T.flip(-1),1,-1)
        torch.testing.assert_close(engine(zero,view='lateral',out_res=16)[0,0],expected,atol=2e-5,rtol=2e-5)

    def test_anisotropic_gaussian_matches_analytic_parallel_integrals(self):
        z,y,x=torch.meshgrid(torch.arange(-16,17)*1.6,torch.arange(-16,17)*1.3,torch.arange(-16,17)*1.,indexing='ij')
        volume=torch.exp(-.5*((x/4)**2+(y/6)**2+(z/5)**2))
        engine=TorchDiffFVREngine(volume,spacing=(1.6,1.3,1.),use_triton=False,detector_spacing=1.)
        v,u=torch.meshgrid(torch.arange(-33,34),torch.arange(-33,34),indexing='ij')
        for view,sigma_u,sigma_ray in [('ap',4.,5.),('lateral',5.,4.)]:
            expected=(2*torch.pi)**.5*sigma_ray*torch.exp(-.5*((u/sigma_u)**2+(v/6.)**2))
            actual=engine(torch.zeros(1,6),view=view,out_res=67)[0,0]
            self.assertLess(float((actual-expected).norm()/expected.norm()),.06)

    def test_parseval_and_six_axis_derivatives(self):
        engine=self.engine()
        true_pose=torch.tensor([[1.,-2.,3.,.3,-.2,.4]])
        targets={view:engine.prepare_spectral_target(engine(true_pose,view=view,out_res=19),high_pass=False).detach()
                 for view in ['ap','lateral']}
        pose=torch.tensor([[7.2,-4.6,3.1,.7,-.4,.6]],requires_grad=True)
        def loss(p):
            return sum(engine.forward_spectral_ncc(p,targets[v],view=v,high_pass=False) for v in targets)/2
        direct=loss(pose)
        reference=[]
        for view,target in targets.items():
            spectrum=engine._extract_slice_spectrum(pose,view=view,out_res=19).unsqueeze(1)
            image=torch.fft.ifft2(torch.fft.ifftshift(spectrum,dim=(-2,-1)))
            target_image=torch.fft.ifft2(torch.fft.ifftshift(target,dim=(-2,-1)))
            dot=(image*target_image.conj()).real.sum((-2,-1))
            norm=(image.abs().square().sum((-2,-1))*target_image.abs().square().sum((-2,-1))).sqrt()
            reference.append(1-(dot/(norm+1e-8/(19*19))).mean())
        spatial=sum(reference)/2
        torch.testing.assert_close(direct,spatial,atol=2e-6,rtol=2e-6)
        gradient=torch.autograd.grad(direct,pose,retain_graph=True)[0]
        other=torch.autograd.grad(spatial,pose)[0]
        torch.testing.assert_close(gradient,other,atol=2e-6,rtol=2e-4)
        self.assertTrue(bool((gradient.abs()>1e-5).all()),'Both independent views must constrain all six parameters on this asymmetric fixture')
        finite=[]
        for axis in range(6):
            step=.01 if axis<3 else .001
            plus=pose.detach().clone();minus=plus.clone()
            plus[0,axis]+=step;minus[0,axis]-=step
            finite.append(float((loss(plus)-loss(minus))/(2*step)))
        finite=torch.tensor([finite])
        self.assertLess(float((finite-gradient).norm()/gradient.norm()),.02)

    def test_invalid_geometry_and_pose_fail_explicitly(self):
        for spacing in [(0,1,1),(1,float('nan'),1),(1,2)]:
            with self.assertRaises(ValueError):
                self.engine(spacing)
        for pitch in [0,float('inf')]:
            with self.assertRaises(ValueError):
                TorchDiffFVREngine(self.volume,detector_spacing=pitch)
        engine=self.engine()
        for pose,view,res in [(torch.zeros(1,5),'ap',19),(torch.full((1,6),float('nan')),'ap',19),
                              (torch.zeros(1,6),'unknown',19),(torch.zeros(1,6),'ap',1)]:
            with self.assertRaises(ValueError):
                engine(pose,view=view,out_res=res)

    def test_true_pose_spectral_objective_has_correct_spatial_origin(self):
        for size in [8,9]:
            volume=torch.zeros(size,size,size);volume[2,3,5]=1.;volume[6,2,3]=2.
            engine=TorchDiffFVREngine(volume,detector_spacing=1.,use_triton=False)
            res=engine.pad_shape[0]
            for view in ['ap','lateral']:
                pose=torch.zeros(1,6,requires_grad=True)
                target=engine(pose.detach(),view=view,out_res=res)
                spectrum=engine.prepare_spectral_target(target,high_pass=False)
                loss=engine.forward_spectral_ncc(pose,spectrum,view=view,high_pass=False)
                self.assertLess(abs(float(loss.detach())),2e-6)
                self.assertLess(float(torch.autograd.grad(loss,pose)[0].abs().max()),2e-5)

    def test_out_of_field_support_is_rejected_instead_of_wrapping(self):
        engine=self.engine()
        with self.assertRaisesRegex(ValueError,'field of view'):
            engine(torch.tensor([[0.,0.,0.,20.,0.,0.]]),out_res=19)
        with self.assertRaisesRegex(ValueError,'field of view'):
            engine(torch.zeros(1,6),out_res=5)

    def test_oblique_fractional_pose_against_closed_form_gaussian(self):
        spacing=(1.1,.9,.7)
        z,y,x=torch.meshgrid(*(torch.arange(-32,33)*a for a in spacing),indexing='ij')
        mean=torch.tensor([12.,-10.,8.]);sigma=torch.tensor([2.5,3.6,2.8])
        volume=torch.exp(-.5*(((torch.stack([x,y,z],-1)-mean)/sigma)**2).sum(-1))
        pose=torch.tensor([[30.,17.,9.,.37,-.23,.45]])
        # Independent standard Euler rotation, not the renderer's helper.
        az,ay,ax=torch.deg2rad(pose[0,:3]);c,sn=torch.cos,torch.sin
        rz=torch.tensor([[c(az),-sn(az),0],[sn(az),c(az),0],[0,0,1.]])
        ry=torch.tensor([[c(ay),0,sn(ay)],[0,1.,0],[-sn(ay),0,c(ay)]])
        rx=torch.tensor([[1.,0,0],[0,c(ax),-sn(ax)],[0,sn(ax),c(ax)]])
        rotation=rz@ry@rx;covariance=torch.diag(sigma.square())
        uv=torch.stack(torch.meshgrid(torch.arange(-50,51),torch.arange(-50,51),indexing='ij')[::-1],-1).float()
        errors={}
        for padding in [2.,4.]:
            engine=TorchDiffFVREngine(volume,spacing=spacing,pad_factor=padding,detector_spacing=1.)
            errors[padding]=[]
            for view,basis in [('ap',torch.tensor([[1.,0,0],[0,1.,0]])),('lateral',torch.tensor([[0.,0,-1.],[0,1.,0]]))]:
                cov=basis@rotation@covariance@rotation.T@basis.T
                center=basis@(rotation@mean+pose[0,3:])
                delta=uv-center
                expected=(2*torch.pi*torch.linalg.det(covariance)/torch.linalg.det(cov)).sqrt()*torch.exp(-.5*torch.einsum('...i,ij,...j->...',delta,torch.linalg.inv(cov),delta))
                actual=engine(pose,view=view,out_res=101)[0,0]
                errors[padding].append(float((actual-expected).norm()/expected.norm()))
        for coarse,fine in zip(errors[2.],errors[4.]):
            self.assertLess(fine,.03,'4x-padding analytic relative L2 bound')
            self.assertLess(fine,coarse,'Padding must reduce the measured interpolation error')
        self.gaussian_diagnostics=errors

    def test_physical_frequency_stencil_and_reduced_precision_fallback(self):
        from src.fvr.triton_slice import fourier_slice_3d_torch,trilinear_slice_analytical_ref
        generator=torch.Generator().manual_seed(207)
        volume=torch.randn(1,2,11,13,15,generator=generator)
        # Physical scaling mimics anisotropic FFT indices; includes out-of-bounds samples.
        u=torch.tensor([[13.1,2.3,-8.7]],requires_grad=True)
        v=torch.tensor([[-4.2,17.3,5.4]],requires_grad=True)
        fx=torch.linspace(-.71,.68,91);fy=torch.linspace(.58,-.63,91)
        reference=trilinear_slice_analytical_ref(volume,u,v,fx,fy,(7.,6.,5.))
        actual=fourier_slice_3d_torch(volume,u,v,fx,fy,(7.,6.,5.))
        torch.testing.assert_close(actual,reference,atol=7e-6,rtol=4e-5)
        a=torch.autograd.grad(actual.square().sum(),(u,v),retain_graph=True)
        b=torch.autograd.grad(reference.square().sum(),(u,v))
        for first,second in zip(a,b):
            torch.testing.assert_close(first,second,atol=2e-4,rtol=2e-4)
        for dtype in [torch.float16,torch.bfloat16]:
            quantized=volume.to(dtype)
            torch.testing.assert_close(fourier_slice_3d_torch(quantized,u,v,fx,fy,(7.,6.,5.)),
                                       fourier_slice_3d_torch(quantized.float(),u,v,fx,fy,(7.,6.,5.)),atol=0,rtol=0)
        # Also exercise the renderer's actual physical bins, guard center and anisotropic axes.
        engine=TorchDiffFVREngine(self.volume,spacing=(1.7,1.3,.9),detector_spacing=.4)
        rotation=engine._euler_to_rotation_matrix(torch.tensor([23.]),torch.tensor([17.]),torch.tensor([9.]))
        depth,height,width=engine.pad_shape
        scale=torch.tensor([width*.9,height*1.3,depth*1.7])
        _,_,ku,kv=engine._get_freq_grid(101,torch.device('cpu'),torch.float32)
        center=(width//2+1,height//2+1,depth//2+1)
        for u_axis,v_axis in [(rotation[:,0,:],rotation[:,1,:]),(-rotation[:,2,:],rotation[:,1,:])]:
            u_axis=(u_axis*scale).detach().requires_grad_();v_axis=(v_axis*scale).detach().requires_grad_()
            actual=fourier_slice_3d_torch(engine.f_vol_5d,u_axis,v_axis,ku,kv,center)
            reference=trilinear_slice_analytical_ref(engine.f_vol_5d,u_axis,v_axis,ku,kv,center)
            torch.testing.assert_close(actual,reference,atol=2e-5,rtol=2e-4)
            first=torch.autograd.grad(actual.square().sum(),(u_axis,v_axis),retain_graph=True)
            second=torch.autograd.grad(reference.square().sum(),(u_axis,v_axis))
            for a,b in zip(first,second):torch.testing.assert_close(a,b,atol=2e-3,rtol=3e-4)
        with self.assertRaisesRegex(ValueError,'float16 range'):
            TorchDiffFVREngine(torch.ones(9,9,9)*100.,dtype=torch.float16)
        engine=TorchDiffFVREngine(torch.ones(9,9,9)*100.)
        with self.assertRaisesRegex(ValueError,'float16 range'):
            engine.to_precision(torch.float16)

    def test_display_projection_gradient_at_second_oblique_pose(self):
        engine=self.engine()
        target={v:engine(torch.tensor([[2.,-3.,1.,.2,-.1,.3]]),view=v,out_res=31).detach() for v in ['ap','lateral']}
        def loss(p):
            return sum((engine(p,view=v,out_res=31)-target[v]).square().mean() for v in target)
        pose=torch.tensor([[-8.3,6.2,-4.7,-.6,.3,-.8]],requires_grad=True)
        gradient=torch.autograd.grad(loss(pose),pose)[0]
        finite=[]
        for axis in range(6):
            step=.01 if axis<3 else .001
            plus=pose.detach().clone();minus=plus.clone();plus[0,axis]+=step;minus[0,axis]-=step
            finite.append(float((loss(plus)-loss(minus))/(2*step)))
        self.assertLess(float((torch.tensor([finite])-gradient).norm()/gradient.norm()),.03)


if __name__=='__main__':
    unittest.main()
