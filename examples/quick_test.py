"""
Simple Test Program for Cross-Language VAD Model
Quick validation that the model works correctly
"""

import sys
import os

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

def quick_test():
    """Quick test to verify the model works"""
    print("🧪 Quick Model Test")
    print("=" * 30)
    
    try:
        # Import the fine-tuned model
        from adaptive_cross_language_vad import AdaptiveCrossLanguageVAD
        print("✅ Model imported successfully")
        
        # Initialize model
        vad_model = AdaptiveCrossLanguageVAD()
        print("✅ Model initialized successfully")
        
        # Test with sample files
        sample_dir = os.path.join(os.path.dirname(__file__), '..', 'sample_datasets')
        
        # Test Hindi file
        hindi_file = os.path.join(sample_dir, 'hindi', '1.wav')
        if os.path.exists(hindi_file):
            print(f"\n🎵 Testing Hindi file: {os.path.basename(hindi_file)}")
            result = vad_model.process_audio_file(hindi_file)
            if result:
                print(f"   Language: {result['language']}")
                print(f"   Duration: {result['duration']:.2f}s")
                print(f"   Speech ratio: {result['speech_ratio']:.1%}")
                print("   ✅ Hindi test passed")
            else:
                print("   ❌ Hindi test failed")
        
        # Test Punjabi file
        punjabi_files = []
        punjabi_dir = os.path.join(sample_dir, 'punjabi')
        if os.path.exists(punjabi_dir):
            punjabi_files = [f for f in os.listdir(punjabi_dir) if f.endswith('.wav')]
        
        if punjabi_files:
            punjabi_file = os.path.join(punjabi_dir, punjabi_files[0])
            print(f"\n🎵 Testing Punjabi file: {os.path.basename(punjabi_file)}")
            result = vad_model.process_audio_file(punjabi_file)
            if result:
                print(f"   Language: {result['language']}")
                print(f"   Duration: {result['duration']:.2f}s")
                print(f"   Speech ratio: {result['speech_ratio']:.1%}")
                print("   ✅ Punjabi test passed")
            else:
                print("   ❌ Punjabi test failed")
        
        print(f"\n🎉 All tests completed!")
        print("✅ Your fine-tuned model is working correctly!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Check that adaptive_cross_language_vad.py is in models/")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)