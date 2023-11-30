import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def results_comparison_in_graph(out_name: str):  
    method = ['NetVLAD', 'Berton et.al', 'Orhan et.al', 'Swin-T', 'PanoVPR(Swin-T)',
                'Swin-S', 'PanoVPR(Swin-S)', 'ConvNeXt-T', 'PanoVPR(ConvNeXt-T)',
                'ConvNeXt-S', 'PanoVPR(ConvNeXt-S)']
    params = [7.23, 86.86, 136.62, 28.29, 28.29, 49.61, 49.61, 28.59, 28.59, 50.22, 50.22]
    r1 = [4.0, 8.0, 47.0, 10.1, 41.4, 12.4, 38.2, 9.7, 34.0, 14.2, 48.8]

    fig, ax = plt.subplots()

    for idx, (m,p,r) in enumerate(zip(method,params,r1)):
        if m == 'NetVLAD':
            ax.scatter(x=p, y=r, s=100, c='#1f77b4', label=m, alpha=0.5)
        elif m == 'Berton et.al':
            ax.scatter(x=p, y=r, s=100, c='#ff7f0e', label=m, alpha=0.5)
        elif m == 'Orhan et.al':
            ax.scatter(x=p, y=r, s=100, c='#2ca02c', label=m, alpha=0.5)
        elif m == 'Swin-T' or m == 'PanoVPR(Swin-T)':
            if m == 'PanoVPR(Swin-T)':
                ax.scatter(x=p, y=r, s=100, c='#d62728', label=m, marker='*', alpha=0.5)
                # add curve
                x_base, y_base = params[idx-1], r1[idx-1]
                x_end, y_end = p, r
                ax.annotate("", xy=(x_end, y_end),xytext=(x_base, y_base),size=4, va="center", ha="center",
                            arrowprops=dict(color='#d62728',
                                            arrowstyle="-|>, head_length=1, head_width=0.4",
                                            linewidth=1,
                                            connectionstyle="arc3,rad=-0.3",
                                            linestyle='dashed',
                            )
                )
            else:
                ax.scatter(x=p, y=r, s=100, c='#d62728', label=m, alpha=0.5)
        elif m == 'Swin-S' or m == 'PanoVPR(Swin-S)':
            if m == 'PanoVPR(Swin-S)':
                ax.scatter(x=p, y=r, s=100, c='#bcbd22', label=m, marker='*', alpha=0.5)
                # add curve
                x_base, y_base = params[idx-1], r1[idx-1]
                x_end, y_end = p, r
                ax.annotate("", xy=(x_end, y_end),xytext=(x_base, y_base),size=4, va="center", ha="center",
                            arrowprops=dict(color='#bcbd22',
                                            arrowstyle="-|>, head_length=1, head_width=0.4",
                                            linewidth=1,
                                            connectionstyle="arc3,rad=-0.2",
                                            linestyle='dashed',
                            )
                )
            else:    
                ax.scatter(x=p, y=r, s=100, c='#bcbd22', label=m, alpha=0.5)
        elif m == 'ConvNeXt-T' or m =='PanoVPR(ConvNeXt-T)':
            if m == 'PanoVPR(ConvNeXt-T)':
                ax.scatter(x=p, y=r, s=100, c='#7f7f7f', label=m, marker='*', alpha=0.5)
                # add curve
                x_base, y_base = params[idx-1], r1[idx-1]
                x_end, y_end = p, r
                ax.annotate("", xy=(x_end, y_end),xytext=(x_base, y_base),size=4, va="center", ha="center",
                            arrowprops=dict(color='#7f7f7f',
                                            arrowstyle="-|>, head_length=1, head_width=0.4",
                                            linewidth=1, 
                                            connectionstyle="arc3,rad=0.2",
                                            linestyle='dashed',
                            )
                )
            else:
                ax.scatter(x=p, y=r, s=100, c='#7f7f7f', label=m, alpha=0.5)
        elif m == 'ConvNeXt-S' or m == 'PanoVPR(ConvNeXt-S)':
            if m == 'PanoVPR(ConvNeXt-S)':
                ax.scatter(x=p, y=r, s=100, c='#e377c2', label=m, marker='*', alpha=0.5)
                # add curve
                x_base, y_base = params[idx-1], r1[idx-1]
                x_end, y_end = p, r
                ax.annotate("", xy=(x_end, y_end),xytext=(x_base, y_base),size=4, va="center", ha="center",
                            arrowprops=dict(color='#e377c2',
                                            arrowstyle="-|>, head_length=1, head_width=0.4",
                                            linewidth=1,
                                            connectionstyle="arc3,rad=0.3",
                                            linestyle='dashed',
                            )
                )
            else:    
                ax.scatter(x=p, y=r, s=100, c='#e377c2', label=m, alpha=0.5)


    # add legend
    ax.legend(loc="center right", prop = {'size':8})

    plt.title('Performance comparison of different methods')
    plt.xlabel('Parameters(M)')
    plt.ylabel('R@1(%)')
    plt.grid(True)

    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.show()


if __name__=="__main__":
    results_comparison_in_graph('test.png')