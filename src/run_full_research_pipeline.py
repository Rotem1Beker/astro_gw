# run_full_research_pipeline.py
# 🚀 פייפליין מחקרי מלא לניתוח GW150914

import os
import sys
import subprocess
import argparse
from pathlib import Path

STEPS = {
    "1": {
        "name": "Event vs Noise Analysis",
        "script": "analyze_event_window.py",
        "description": "מנתח את חלון האירוע לעומת רעש רקע, כולל spectrograms ופוסטריורים"
    },
    "2": {
        "name": "Bayes Factor Calculation",
        "script": "bayes_factor.py",
        "description": "מחשב Bayes Factor מפורט להשוואת signal vs noise hypotheses"
    },
    "3": {
        "name": "Test Realistic Waveforms",
        "script": "train_realistic_waveforms.py --test",
        "description": "בודק יצירת waveforms ריאליסטיים (chirp signals)"
    },
    "4": {
        "name": "Train with Realistic Waveforms",
        "script": "train_realistic_waveforms.py --train",
        "description": "אימון מחדש של המודל עם waveforms ריאליסטיים במקום toy models"
    },
    "5": {
        "name": "Re-run Inference with New Model",
        "script": "infer_online.py --model model_realistic.pt",
        "description": "מריץ inference מחדש עם המודל המאומן על waveforms ריאליסטיים"
    }
}


def print_banner():
    print("\n" + "="*70)
    print("🌊 GRAVITATIONAL WAVE DETECTION RESEARCH PIPELINE")
    print("="*70)
    print("Analyzing LIGO GW150914 event with Neural Posterior Estimation")
    print("="*70 + "\n")


def print_menu():
    print("\n📋 Available Steps:")
    print("-" * 70)
    for key, step in STEPS.items():
        print(f"  [{key}] {step['name']}")
        print(f"      → {step['description']}")
        print()
    print("  [A] Run ALL steps sequentially")
    print("  [Q] Quit")
    print("-" * 70)


def run_step(step_num):
    """מריץ צעד מסוים"""
    if step_num not in STEPS:
        print(f"❌ Invalid step: {step_num}")
        return False
    
    step = STEPS[step_num]
    print(f"\n{'='*70}")
    print(f"🚀 Running Step {step_num}: {step['name']}")
    print(f"{'='*70}\n")
    
    # בדיקת קיום הסקריפט
    script_name = step['script'].split()[0]
    if not Path(script_name).exists():
        print(f"⚠️  Script not found: {script_name}")
        print(f"   Please make sure the file exists in the current directory")
        return False
    
    # הרצה
    try:
        if "--test" in step['script']:
            # רק בדיקת waveforms
            subprocess.run([sys.executable, "train_realistic_waveforms.py"], 
                         check=True, env={**os.environ, "TEST_ONLY": "1"})
        elif "--train" in step['script']:
            # אימון מלא
            subprocess.run([sys.executable, "train_realistic_waveforms.py"], 
                         check=True)
        else:
            # הרצה רגילה
            subprocess.run([sys.executable, script_name], check=True)
        
        print(f"\n✅ Step {step_num} completed successfully!\n")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Step {step_num} failed with error code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Python interpreter or script not found")
        return False


def run_all_steps():
    """מריץ את כל הצעדים ברצף"""
    print("\n" + "🎬 " * 20)
    print("RUNNING FULL RESEARCH PIPELINE")
    print("🎬 " * 20 + "\n")
    
    results = {}
    for step_num in sorted(STEPS.keys()):
        success = run_step(step_num)
        results[step_num] = success
        
        if not success:
            print(f"\n⚠️  Pipeline stopped at step {step_num}")
            print("You can continue from this step later")
            break
        
        # הפסקה קצרה בין צעדים
        input("\n⏸️  Press Enter to continue to next step...")
    
    # סיכום
    print("\n" + "="*70)
    print("📊 PIPELINE SUMMARY")
    print("="*70)
    for step_num, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  Step {step_num}: {STEPS[step_num]['name']:<40} {status}")
    print("="*70 + "\n")


def check_requirements():
    """בדיקת דרישות בסיסיות"""
    print("🔍 Checking requirements...")
    
    required_files = [
        "config.yaml",
        "model.pt",
        "data/H-H1_LOSC_4_V1-1126259446-32.hdf5"
    ]
    
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
    
    if missing:
        print("⚠️  Missing required files:")
        for file in missing:
            print(f"   - {file}")
        print("\nPlease make sure these files exist before running the pipeline.")
        return False
    
    print("✅ All required files found")
    return True


def interactive_mode():
    """מצב אינטראקטיבי"""
    print_banner()
    
    if not check_requirements():
        return
    
    while True:
        print_menu()
        choice = input("\n👉 Select option: ").strip().upper()
        
        if choice == 'Q':
            print("\n👋 Goodbye!\n")
            break
        elif choice == 'A':
            run_all_steps()
        elif choice in STEPS:
            run_step(choice)
            input("\n⏸️  Press Enter to return to menu...")
        else:
            print(f"❌ Invalid choice: {choice}")


def main():
    parser = argparse.ArgumentParser(
        description="Gravitational Wave Detection Research Pipeline"
    )
    parser.add_argument(
        "--step",
        type=str,
        help="Run specific step (1-5, or 'all')"
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run without interactive prompts"
    )
    
    args = parser.parse_args()
    
    if args.step:
        if not check_requirements():
            return
        
        if args.step.lower() == 'all':
            run_all_steps()
        elif args.step in STEPS:
            run_step(args.step)
        else:
            print(f"❌ Invalid step: {args.step}")
            print(f"Available steps: {', '.join(STEPS.keys())}, all")
    else:
        interactive_mode()


if __name__ == "__main__":
    main()