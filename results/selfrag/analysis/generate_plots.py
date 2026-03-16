import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['figure.dpi'] = 300

os.makedirs('/home/sbhengesa/selfrag/analysis/figures', exist_ok=True)

datasets_info = {
    'PopQA': 'popqa',
    'TriviaQA': 'triviaqa',
    'PubHealth': 'pubhealth',
    'ARC-C': 'arc'
}

results = {}
for name, prefix in datasets_info.items():
    results[name] = {}
    for mode in ['adaptive', 'always_retrieve', 'no_retrieval']:
        with open(f'/home/sbhengesa/selfrag/results/{prefix}_{mode}.json') as f:
            results[name][mode] = json.load(f)

datasets = list(datasets_info.keys())
adaptive = [results[d]['adaptive']['metric_mean'] * 100 for d in datasets]
always_ret = [results[d]['always_retrieve']['metric_mean'] * 100 for d in datasets]
no_ret = [results[d]['no_retrieval']['metric_mean'] * 100 for d in datasets]

retrieval_freq = {}
for name in datasets:
    data = results[name]['adaptive']
    total = len(data['all_results'])
    retrieved = sum(1 for r in data['all_results'] if len(r) > 1)
    retrieval_freq[name] = {'total': total, 'retrieved': retrieved, 'pct': retrieved / total * 100}

# ============================================================
# FIGURE 1: Accuracy comparison grouped bar chart
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(datasets))
width = 0.25

bars1 = ax.bar(x - width, adaptive, width, label='Adaptive retrieval', color='#2563EB', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x, always_ret, width, label='Always retrieve', color='#059669', edgecolor='white', linewidth=0.5)
bars3 = ax.bar(x + width, no_ret, width, label='No retrieval', color='#DC2626', edgecolor='white', linewidth=0.5)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Accuracy (%)', fontsize=13)
ax.set_title('Self-RAG: accuracy across retrieval modes', fontsize=15, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=12)
ax.set_ylim(0, 85)
ax.legend(loc='upper right', frameon=True, edgecolor='#E5E7EB', fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('/home/sbhengesa/selfrag/analysis/figures/fig1_accuracy_comparison.png', bbox_inches='tight')
plt.savefig('/home/sbhengesa/selfrag/analysis/figures/fig1_accuracy_comparison.pdf', bbox_inches='tight')
plt.close()
print("Figure 1 saved")

# ============================================================
# FIGURE 2: Retrieval frequency horizontal bar
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
datasets_rev = list(reversed(datasets))
retrieved_pct = [retrieval_freq[d]['pct'] for d in datasets_rev]
skipped_pct = [100 - retrieval_freq[d]['pct'] for d in datasets_rev]
y = np.arange(len(datasets_rev))

ax.barh(y, retrieved_pct, height=0.5, label='Retrieved', color='#7C3AED', edgecolor='white', linewidth=0.5)
ax.barh(y, skipped_pct, height=0.5, left=retrieved_pct, label='Skipped retrieval', color='#E5E7EB', edgecolor='white', linewidth=0.5)

for i, (ret, skip) in enumerate(zip(retrieved_pct, skipped_pct)):
    ax.text(ret / 2, i, f'{ret:.1f}%', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    if skip > 2:
        ax.text(ret + skip / 2, i, f'{skip:.1f}%', ha='center', va='center', fontsize=11, fontweight='bold', color='#6B7280')

ax.set_yticks(y)
ax.set_yticklabels(datasets_rev, fontsize=12)
ax.set_xlabel('Percentage of queries (%)', fontsize=13)
ax.set_title('Self-RAG retrieval frequency in adaptive mode', fontsize=15, fontweight='bold', pad=15)
ax.set_xlim(0, 105)
ax.legend(loc='lower right', frameon=True, edgecolor='#E5E7EB', fontsize=11)
ax.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('/home/sbhengesa/selfrag/analysis/figures/fig2_retrieval_frequency.png', bbox_inches='tight')
plt.savefig('/home/sbhengesa/selfrag/analysis/figures/fig2_retrieval_frequency.pdf', bbox_inches='tight')
plt.close()
print("Figure 2 saved")

# ============================================================
# FIGURE 3: Retrieval gain positive/negative bar
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
gains = [adaptive[i] - no_ret[i] for i in range(len(datasets))]
colors = ['#059669' if g >= 0 else '#DC2626' for g in gains]

bars = ax.bar(datasets, gains, color=colors, width=0.5, edgecolor='white', linewidth=0.5)
for bar, gain in zip(bars, gains):
    ax.annotate(f'{gain:+.1f} pp', xy=(bar.get_x() + bar.get_width() / 2, gain),
                xytext=(0, 6 if gain >= 0 else -16), textcoords="offset points",
                ha='center', va='bottom' if gain >= 0 else 'top',
                fontsize=11, fontweight='bold', color='#059669' if gain >= 0 else '#DC2626')

ax.axhline(y=0, color='#9CA3AF', linewidth=0.8)
ax.set_ylabel('Accuracy change (percentage points)', fontsize=13)
ax.set_title('Impact of retrieval on accuracy', fontsize=15, fontweight='bold', pad=15)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(min(gains) - 5, max(gains) + 5)
plt.tight_layout()
plt.savefig('/home/sbhengesa/selfrag/analysis/figures/fig3_retrieval_gain.png', bbox_inches='tight')
plt.savefig('/home/sbhengesa/selfrag/analysis/figures/fig3_retrieval_gain.pdf', bbox_inches='tight')
plt.close()
print("Figure 3 saved")

# ============================================================
# FIGURE 4: Decision effectiveness quadrant scatter
# ============================================================
fig, ax = plt.subplots(figsize=(8, 7))
for i, name in enumerate(datasets):
    ret_pct = retrieval_freq[name]['pct']
    gain = adaptive[i] - no_ret[i]
    color = '#059669' if gain > 1 else ('#DC2626' if gain < -0.5 else '#F59E0B')
    size = retrieval_freq[name]['total'] / 8
    ax.scatter(ret_pct, gain, s=size, c=color, alpha=0.8, edgecolors='white', linewidth=1.5, zorder=5)
    ax.annotate(name, (ret_pct, gain), textcoords="offset points", xytext=(12, 0), fontsize=12, fontweight='bold')

ax.axhline(y=0, color='#9CA3AF', linewidth=0.8, linestyle='--')
ax.axvline(x=50, color='#9CA3AF', linewidth=0.8, linestyle='--')
ax.text(75, max(gains) * 0.85, 'Retrieves often\n& it helps', ha='center', fontsize=10, color='#059669', fontstyle='italic', alpha=0.7)
ax.text(75, -2, 'Retrieves often\nbut unnecessary', ha='center', fontsize=10, color='#DC2626', fontstyle='italic', alpha=0.7)
ax.set_xlabel('Retrieval frequency (%)', fontsize=13)
ax.set_ylabel('Accuracy gain from retrieval (pp)', fontsize=13)
ax.set_title('Retrieval decision effectiveness', fontsize=15, fontweight='bold', pad=15)
ax.set_xlim(-5, 110)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('/home/sbhengesa/selfrag/analysis/figures/fig4_decision_effectiveness.png', bbox_inches='tight')
plt.savefig('/home/sbhengesa/selfrag/analysis/figures/fig4_decision_effectiveness.pdf', bbox_inches='tight')
plt.close()
print("Figure 4 saved")

# ============================================================
# FIGURE 5: Per-dataset subplot grid
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for idx, name in enumerate(datasets):
    ax = axes[idx]
    modes = ['Adaptive', 'Always\nretrieve', 'No\nretrieval']
    accs = [results[name]['adaptive']['metric_mean'] * 100, results[name]['always_retrieve']['metric_mean'] * 100, results[name]['no_retrieval']['metric_mean'] * 100]
    clrs = ['#2563EB', '#059669', '#DC2626']
    bars = ax.bar(modes, accs, color=clrs, width=0.6, edgecolor='white', linewidth=0.5)
    for bar, acc in zip(bars, accs):
        ax.annotate(f'{acc:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, acc),
                    xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=11, fontweight='bold')
    total = retrieval_freq[name]['total']
    ret_pct = retrieval_freq[name]['pct']
    ax.set_title(f'{name}  (n={total}, ret={ret_pct:.1f}%)', fontsize=13, fontweight='bold', pad=10)
    ax.set_ylim(0, max(accs) + 12)
    ax.grid(axis='y', alpha=0.2, linestyle='--')
fig.suptitle('Self-RAG baseline results by dataset', fontsize=17, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/home/sbhengesa/selfrag/analysis/figures/fig5_per_dataset_breakdown.png', bbox_inches='tight')
plt.savefig('/home/sbhengesa/selfrag/analysis/figures/fig5_per_dataset_breakdown.pdf', bbox_inches='tight')
plt.close()
print("Figure 5 saved")

# ============================================================
# TABLE 1: Main results (saved as formatted text + CSV)
# ============================================================
print("\n" + "="*75)
print("TABLE 1: MAIN RESULTS")
print("="*75)
header = f"{'Dataset':<15} {'Task Type':<18} {'Adaptive':<12} {'Always Ret':<12} {'No Ret':<12}"
print(header)
print("-" * 69)
task_types = {'PopQA': 'Long-tail QA', 'TriviaQA': 'Open-domain QA', 'PubHealth': 'Fact verification', 'ARC-C': 'Science reasoning'}
for name in datasets:
    a = results[name]['adaptive']['metric_mean'] * 100
    ar = results[name]['always_retrieve']['metric_mean'] * 100
    nr = results[name]['no_retrieval']['metric_mean'] * 100
    print(f"{name:<15} {task_types[name]:<18} {a:<12.1f} {ar:<12.1f} {nr:<12.1f}")

# Save as CSV
with open('/home/sbhengesa/selfrag/analysis/figures/table1_main_results.csv', 'w') as f:
    f.write("Dataset,Task Type,Adaptive,Always Retrieve,No Retrieval\n")
    for name in datasets:
        a = results[name]['adaptive']['metric_mean'] * 100
        ar = results[name]['always_retrieve']['metric_mean'] * 100
        nr = results[name]['no_retrieval']['metric_mean'] * 100
        f.write(f"{name},{task_types[name]},{a:.1f},{ar:.1f},{nr:.1f}\n")

# ============================================================
# TABLE 2: Retrieval decision analysis
# ============================================================
print("\n" + "="*75)
print("TABLE 2: RETRIEVAL DECISION ANALYSIS")
print("="*75)
header = f"{'Dataset':<15} {'Total':<8} {'Retrieved':<12} {'Skipped':<10} {'Ret %':<8} {'Gain (pp)':<10} {'Decision':<15}"
print(header)
print("-" * 78)
for i, name in enumerate(datasets):
    total = retrieval_freq[name]['total']
    retrieved = retrieval_freq[name]['retrieved']
    skipped = total - retrieved
    ret_pct = retrieval_freq[name]['pct']
    gain = adaptive[i] - no_ret[i]
    if gain > 2:
        decision = "Correct (helps)"
    elif gain < -0.5:
        decision = "Wrong (hurts)"
    else:
        decision = "Neutral"
    print(f"{name:<15} {total:<8} {retrieved:<12} {skipped:<10} {ret_pct:<8.1f} {gain:<+10.1f} {decision:<15}")

with open('/home/sbhengesa/selfrag/analysis/figures/table2_retrieval_decision.csv', 'w') as f:
    f.write("Dataset,Total Questions,Retrieved,Skipped,Retrieval %,Accuracy Gain (pp),Decision Quality\n")
    for i, name in enumerate(datasets):
        total = retrieval_freq[name]['total']
        retrieved = retrieval_freq[name]['retrieved']
        skipped = total - retrieved
        ret_pct = retrieval_freq[name]['pct']
        gain = adaptive[i] - no_ret[i]
        decision = "Correct" if gain > 2 else ("Wrong" if gain < -0.5 else "Neutral")
        f.write(f"{name},{total},{retrieved},{skipped},{ret_pct:.1f},{gain:+.1f},{decision}\n")

# ============================================================
# TABLE 3: Error analysis
# ============================================================
print("\n" + "="*75)
print("TABLE 3: ERROR ANALYSIS")
print("="*75)
header = f"{'Dataset':<15} {'Correct':<10} {'Wrong':<10} {'Correct w/':<12} {'Wrong w/':<12} {'Unnecessary':<12}"
print(f"{'':15} {'(adapt)':<10} {'(adapt)':<10} {'no ret too':<12} {'but no ret ok':<12} {'retrievals':<12}")
print("-" * 71)

for i, name in enumerate(datasets):
    adapt_results = results[name]['adaptive']['metric_results']
    noret_results = results[name]['no_retrieval']['metric_results']
    
    total = len(adapt_results)
    adapt_correct = sum(adapt_results)
    adapt_wrong = total - adapt_correct
    
    both_correct = sum(1 for a, n in zip(adapt_results, noret_results) if a == 1 and n == 1)
    adapt_wrong_noret_correct = sum(1 for a, n in zip(adapt_results, noret_results) if a == 0 and n == 1)
    adapt_correct_noret_wrong = sum(1 for a, n in zip(adapt_results, noret_results) if a == 1 and n == 0)
    both_wrong = sum(1 for a, n in zip(adapt_results, noret_results) if a == 0 and n == 0)
    
    unnecessary = both_correct
    
    print(f"{name:<15} {adapt_correct:<10} {adapt_wrong:<10} {both_correct:<12} {adapt_wrong_noret_correct:<12} {both_correct:<12}")

with open('/home/sbhengesa/selfrag/analysis/figures/table3_error_analysis.csv', 'w') as f:
    f.write("Dataset,Adaptive Correct,Adaptive Wrong,Both Correct,Adaptive Wrong but NoRet Correct,Adaptive Correct but NoRet Wrong,Both Wrong\n")
    for i, name in enumerate(datasets):
        adapt_results = results[name]['adaptive']['metric_results']
        noret_results = results[name]['no_retrieval']['metric_results']
        total = len(adapt_results)
        both_correct = sum(1 for a, n in zip(adapt_results, noret_results) if a == 1 and n == 1)
        adapt_wrong_noret_correct = sum(1 for a, n in zip(adapt_results, noret_results) if a == 0 and n == 1)
        adapt_correct_noret_wrong = sum(1 for a, n in zip(adapt_results, noret_results) if a == 1 and n == 0)
        both_wrong = sum(1 for a, n in zip(adapt_results, noret_results) if a == 0 and n == 0)
        f.write(f"{name},{sum(adapt_results)},{total - sum(adapt_results)},{both_correct},{adapt_wrong_noret_correct},{adapt_correct_noret_wrong},{both_wrong}\n")

print(f"\nAll figures and tables saved to: /home/sbhengesa/selfrag/analysis/figures/")
print("DONE!")
