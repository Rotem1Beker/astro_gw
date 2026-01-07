# bayes_factor.py
# חישוב Bayes Factor מדויק יותר להשוואת signal vs noise hypotheses

import numpy as np
from scipy import stats
from scipy.special import logsumexp

class BayesFactorCalculator:
    """
    מחשב Bayes Factor: BF = P(data | signal) / P(data | noise)
    
    גישה: משווים את ה-log-likelihood של הדגימות הפוסטריוריות
    תחת שתי ההיפותזות
    """
    
    def __init__(self, signal_samples, noise_samples):
        """
        Args:
            signal_samples: (n_samples, n_params) דגימות מהפוסטריור עם signal
            noise_samples: (n_samples, n_params) דגימות מהפוסטריור עם noise
        """
        self.signal_samples = signal_samples
        self.noise_samples = noise_samples
        self.n_params = signal_samples.shape[1]
    
    def evidence_via_importance_sampling(self, samples):
        """
        מעריך evidence באמצעות importance sampling
        log P(data) ≈ log(1/N * Σ p(θ)/q(θ))
        
        כאן q(θ) זו ההתפלגות שממנה דגמנו (הפוסטריור),
        ו-p(θ) זו ה-prior
        """
        # נניח prior גאוסי רחב
        prior_mean = np.zeros(self.n_params)
        prior_cov = 100 * np.eye(self.n_params)
        
        # חשב log-likelihood תחת prior
        log_prior_probs = stats.multivariate_normal.logpdf(
            samples, mean=prior_mean, cov=prior_cov
        )
        
        # חשב log-likelihood תחת הפוסטריור (proposal)
        post_mean = samples.mean(axis=0)
        post_cov = np.cov(samples.T) + 1e-6 * np.eye(self.n_params)
        log_proposal_probs = stats.multivariate_normal.logpdf(
            samples, mean=post_mean, cov=post_cov
        )
        
        # Importance weights: w = p(θ) / q(θ)
        log_weights = log_prior_probs - log_proposal_probs
        
        # Log evidence: log(1/N * Σ exp(log_weights))
        log_evidence = logsumexp(log_weights) - np.log(len(samples))
        
        return log_evidence
    
    def gaussian_approximation(self, samples):
        """
        הערכה פשוטה יותר: נניח שהפוסטריור גאוסי
        log P(data) ≈ -0.5 * log(det(Σ)) - const
        
        פוסטריור מרוכז יותר → evidence גבוה יותר
        """
        cov = np.cov(samples.T) + 1e-6 * np.eye(self.n_params)
        log_det = np.linalg.slogdet(cov)[1]
        
        # Evidence יורד עם נפח הפוסטריור
        return -0.5 * log_det
    
    def kl_divergence_method(self):
        """
        מודד כמה שונים הפוסטריורים: KL(signal || noise)
        אם הם דומים → אין סיגנל
        אם שונים → יש סיגנל
        """
        signal_mean = self.signal_samples.mean(axis=0)
        signal_cov = np.cov(self.signal_samples.T) + 1e-6 * np.eye(self.n_params)
        
        noise_mean = self.noise_samples.mean(axis=0)
        noise_cov = np.cov(self.noise_samples.T) + 1e-6 * np.eye(self.n_params)
        
        # KL divergence בין שתי התפלגויות גאוסיות
        inv_noise_cov = np.linalg.inv(noise_cov)
        
        kl = 0.5 * (
            np.trace(inv_noise_cov @ signal_cov) +
            (noise_mean - signal_mean).T @ inv_noise_cov @ (noise_mean - signal_mean) -
            self.n_params +
            np.linalg.slogdet(noise_cov)[1] - np.linalg.slogdet(signal_cov)[1]
        )
        
        return kl
    
    def compute_all_metrics(self):
        """מחשב את כל המטריקות"""
        results = {
            # Evidence ratio (Bayes Factor)
            'log_bf_importance_sampling': (
                self.evidence_via_importance_sampling(self.signal_samples) -
                self.evidence_via_importance_sampling(self.noise_samples)
            ),
            
            'log_bf_gaussian': (
                self.gaussian_approximation(self.signal_samples) -
                self.gaussian_approximation(self.noise_samples)
            ),
            
            # KL divergence (מודד שוני בין posteriors)
            'kl_divergence': self.kl_divergence_method(),
            
            # סטטיסטיקות נוספות
            'signal_posterior_volume': np.linalg.det(
                np.cov(self.signal_samples.T) + 1e-6 * np.eye(self.n_params)
            ),
            'noise_posterior_volume': np.linalg.det(
                np.cov(self.noise_samples.T) + 1e-6 * np.eye(self.n_params)
            ),
            'distance_between_means': np.linalg.norm(
                self.signal_samples.mean(axis=0) - self.noise_samples.mean(axis=0)
            )
        }
        
        return results
    
    def interpret_results(self, results):
        """פירוש התוצאות"""
        print("\n" + "="*60)
        print("📊 BAYES FACTOR ANALYSIS")
        print("="*60)
        
        print(f"\n🎯 Log Bayes Factor (importance sampling): {results['log_bf_importance_sampling']:.2f}")
        print(f"🎯 Log Bayes Factor (Gaussian approx):     {results['log_bf_gaussian']:.2f}")
        
        # פירוש לפי Jeffrey's scale
        log_bf = results['log_bf_gaussian']
        if log_bf > 5:
            strength = "STRONG evidence for signal"
        elif log_bf > 3:
            strength = "Substantial evidence for signal"
        elif log_bf > 1:
            strength = "Weak evidence for signal"
        elif log_bf > -1:
            strength = "Inconclusive"
        elif log_bf > -3:
            strength = "Weak evidence for noise-only"
        else:
            strength = "Substantial evidence for noise-only"
        
        print(f"   → Interpretation: {strength}")
        print(f"   → BF = {np.exp(log_bf):.2e} (linear scale)")
        
        print(f"\n📏 KL Divergence (signal || noise): {results['kl_divergence']:.4f}")
        print(f"   → Higher = more different posteriors")
        
        print(f"\n📦 Posterior volumes:")
        print(f"   Signal: {results['signal_posterior_volume']:.2e}")
        print(f"   Noise:  {results['noise_posterior_volume']:.2e}")
        print(f"   Ratio:  {results['signal_posterior_volume']/results['noise_posterior_volume']:.2f}")
        
        print(f"\n📍 Distance between posterior means: {results['distance_between_means']:.4f}")
        
        print("="*60 + "\n")
        
        return strength


def calculate_bayes_factor(signal_samples, noise_samples, verbose=True):
    """
    פונקציה נוחה לחישוב Bayes Factor
    
    Args:
        signal_samples: דגימות מפוסטריור עם signal
        noise_samples: דגימות מפוסטריור עם noise
        verbose: האם להדפיס תוצאות
    
    Returns:
        dict עם כל המטריקות
    """
    calc = BayesFactorCalculator(signal_samples, noise_samples)
    results = calc.compute_all_metrics()
    
    if verbose:
        calc.interpret_results(results)
    
    return results


# דוגמה לשימוש
if __name__ == "__main__":
    # סימולציה: signal posterior מרוכז, noise posterior מפוזר
    np.random.seed(42)
    
    signal_samples = np.random.randn(1000, 2) * 0.5 + np.array([10, 20])
    noise_samples = np.random.randn(1000, 2) * 2.0 + np.array([5, 15])
    
    results = calculate_bayes_factor(signal_samples, noise_samples)