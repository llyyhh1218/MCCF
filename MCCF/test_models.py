import sys
import os


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    print("\n" + "=" * 70)
    print("  TEST 1: Import All Modules")
    print("=" * 70)
    
    try:

        from models import (

            SS2D, VSSBlock, Mlp,
            
            ChannelRectifyModule, ChannelWeights, SS1D, ConMB_SS2D,
            
            VSSBlock_Cross_new, Cross_layer, SS2D_cross_new, eca_layer,
            
            VSSM_Fusion_Complete, create_vssm_tiny,
            
            PatchEmbed2D, PatchMerging2D,
        )
        
        print("[OK] All modules imported successfully!")
        print(f"    - Core: SS2D, VSSBlock, Mlp")
        print(f"    - Fusion Stage 1-2: ChannelRectifyModule, ConMB_SS2D")
        print(f"    - Fusion Stage 3: VSSBlock_Cross_new (from FusionMamba-main)")
        print(f"    - Architecture: VSSM_Fusion_Complete")
        print(f"    - Utils: PatchEmbed2D, etc.")
        
        return True
        
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fusion_components():
    print("\n" + "=" * 70)
    print("  TEST 2: Test Fusion Components")
    print("=" * 70)
    
    import torch
    
    try:
        from models.fusion import (
            ChannelRectifyModule,
            ConMB_SS2D,
            VSSBlock_Cross_new,
        )
        
        B, C, H, W = 2, 96, 32, 32
        x1 = torch.randn(B, C, H, W)
        x2 = torch.randn(B, C, H, W)
        
        print("\n[Testing] Stage 1: ChannelRectifyModule...")
        crm = ChannelRectifyModule(dim=C, HW=H*W)
        crm_out1, crm_out2 = crm(x1, x2)
        assert crm_out1.shape == (B, C, H, W), f"CRM output shape mismatch: {crm_out1.shape}"
        print(f"    [OK] CRM output shape: {crm_out1.shape}")
        
        print("\n[Testing] Stage 2: ConMB_SS2D...")
        conmb = ConMB_SS2D(d_model=C)
        conmb_out1, conmb_out2 = conmb(crm_out1.permute(0, 2, 3, 1), 
                                       crm_out2.permute(0, 2, 3, 1))
        assert conmb_out1.shape == (B, H, W, C), f"ConMB output shape mismatch: {conmb_out1.shape}"
        print(f"    [OK] ConMB_SS2D output shape: {conmb_out1.shape}")

        print("\n[Testing] Stage 3: VSSBlock_Cross_new (from FusionMamba-main)...")
        cross_block = VSSBlock_Cross_new(hidden_dim=C)
        fused_output = cross_block(conmb_out1, conmb_out2)
        assert fused_output.shape == (B, H, W, C), f"Cross fusion shape mismatch: {fused_output.shape}"
        print(f"    [OK] VSSBlock_Cross_new output shape: {fused_output.shape}")
        
        print("\n[SUCCESS] All three-stage fusion components work correctly!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_complete_model():

    print("\n" + "=" * 70)
    print("  TEST 3: Complete Model Forward Pass")
    print("=" * 70)
    
    import torch
    
    try:
        from models import create_vssm_tiny
        
        print("\n[Creating] VSSM_Tiny model...")
        model = create_vssm_tiny(
            num_classes=2,     
            in_chans=1,         
            img_size=[256, 256]
        )
        

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"    [OK] Model created successfully!")
        print(f"    Total parameters: {total_params / 1e6:.2f}M")
        print(f"    Trainable parameters: {trainable_params / 1e6:.2f}M")
        
        print("\n[Running] Forward pass with sample inputs...")
        B = 2  # batch size
        x1 = torch.randn(B, 1, 256, 256)  
        x2 = torch.randn(B, 1, 256, 256)  
        

        model.eval()
        
        with torch.no_grad():
            output = model(x1, x2)
        
        expected_shape = (B, 2, 64, 64)  
        assert output.shape == expected_shape, f"Output shape mismatch! Expected {expected_shape}, got {output.shape}"
        
        print(f"    [OK] Forward pass successful!")
        print(f"    Input shapes: x1={x1.shape}, x2={x2.shape}")
        print(f"    Output shape: {output.shape}")
        
        if hasattr(model, 'get_model_info'):
            info = model.get_model_info()
            if 'fusion_stages' in info:
                print(f"\n[Fusion Pipeline]")
                for stage, desc in info['fusion_stages'].items():
                    print(f"    {stage}: {desc}")
        
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Complete model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_structure():
    print("\n" + "=" * 70)
    print("  TEST 4: Verify Model Structure")
    print("=" * 70)
    
    try:
        from models import create_vssm_tiny
        import torch
        
        model = create_vssm_tiny(num_classes=2, img_size=[256, 256])
        
        checks = []
        
        has_crm = hasattr(model, 'CRM_modules')
        has_conmb = hasattr(model, 'ConMB_modules')
        has_cross = hasattr(model, 'Cross_fusion_modules')
        
        checks.append(("Has CRM_modules (Stage 1)", has_crm))
        checks.append(("Has ConMB_modules (Stage 2)", has_conmb))
        checks.append(("Has Cross_fusion_modules (Stage 3)", has_cross))
        
        if has_crm:
            num_crm = len(model.CRM_modules)
            checks.append((f"CRM modules count: {num_crm}", num_crm > 0))
        
        if has_conmb:
            num_conmb = len(model.ConMB_modules)
            checks.append((f"ConMB modules count: {num_conmb}", num_conmb > 0))
        
        if has_cross:
            num_cross = len(model.Cross_fusion_modules)
            checks.append((f"Cross fusion modules count: {num_cross}", num_cross > 0))
        
        all_passed = True
        for name, result in checks:
            status = "[OK]" if result else "[FAIL]"
            print(f"{status} {name}")
            all_passed = all_passed and result
        
        if all_passed:
            print("\n[SUCCESS] Model structure is correct!")
        else:
            print("\n[WARNING] Some structure checks failed")
        
        return all_passed
        
    except Exception as e:
        print(f"[FAIL] Structure verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    
    print("\n" + "=" * 70)
    print("  MCCF Models Integration Test")
    print("=" * 70)
    
    results = []
    
    results.append(("Import Modules", test_imports()))
    results.append(("Fusion Components", test_fusion_components()))
    results.append(("Complete Model", test_complete_model()))
    results.append(("Model Structure", test_model_structure()))
    
    print("\n" + "=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = ">>>" if result else "!!!"
        print(f"{symbol} {name}: {status}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "=" * 70)
        print("  SUCCESS! All tests passed!")
        print("=" * 70)
        print("=" * 70)
    else:
        print("\n[WARNING] Some tests failed. Please check the errors above.")
    
    print("\n")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
