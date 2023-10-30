import io

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from cycler import cycler

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.size": 27,
    "lines.markersize": 5,
})
default_cycler = cycler(color=["#A30000"]) + cycler(linestyle=["-"])

plt.rc("lines", linewidth=4)
plt.rc("lines", linewidth=4)
# plt.rc("axes", prop_cycle=default_cycler)


def plot_to_numpy(fig):
    with io.BytesIO() as buff:
        fig.savefig(buff, format="raw")
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    return data.reshape((int(h), int(w), -1))


v_max = 1.2
n_steps = 100
n_steps_ = 5
discretization = np.linspace(-v_max, v_max, n_steps)
discretization_ = np.linspace(-v_max, v_max, n_steps_)
xx, yy = np.meshgrid(*(2 * (discretization, )))
xy = np.stack((xx, yy), axis=-1).reshape(n_steps**2, 2)
xx_, yy_ = np.meshgrid(*(2 * (discretization_, )))
xy_ = np.stack((xx_, yy_), axis=-1).reshape(n_steps_**2, 2)
ts = np.linspace(0, 0.3, 200)


def generate_data():
    n_points = 6
    rng = np.random.default_rng()
    points = rng.uniform(low=-1.0, high=1.0, size=(n_points, 2))
    weights = rng.uniform(low=0.2, high=1, size=(n_points, ))
    np.save("./points.npy", points)
    np.save("./weights.npy", weights)


def gaussian(x, mu, sigma):
    k = mu.shape[0]
    d = (x - mu[None]).T
    return (
        np.exp(-(d * (np.linalg.inv(sigma) @ d)).sum(0) / 2) /
        np.linalg.det(2 * np.pi * sigma)**0.5
    )


def smooth(grid, mus, weights, t):
    rv = np.zeros(grid.shape[:1])
    for mu, w in zip(mus, weights):
        rv += w * gaussian(grid, mu, t)
    return rv


def nabla_log_smooth(grid, mus, weights, t):
    rv = np.zeros((mus.shape[1], *grid.shape[:1]))
    for mu, w in zip(mus, weights):
        rv += ((np.linalg.inv(t) @ (grid - mu[None]).T) * w *
               gaussian(grid, mu, t)[None])
    return rv / smooth(grid, mus, weights, t)


def vis1(points, weights):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 10))
    _ = ax.stem(
        points[:, 0], points[:, 1], weights, markerfmt="o", basefmt=" "
    )
    ax.set_xlim([-v_max, v_max])
    ax.set_ylim([-v_max, v_max])
    ax.set_zticklabels([])
    ax.set_proj_type("ortho")
    plt.tight_layout()
    plt.savefig("empirical.pdf")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(points[:, 0], points[:, 1], s=weights * 1000)
    ax.set_xlim([-v_max, v_max])
    ax.set_ylim([-v_max, v_max])
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig("empirical_2d.pdf")


def style(axes):
    for axis in axes:
        axis.set_ticklabels([])
        axis._axinfo["axisline"]["linewidth"] = 1
        axis._axinfo["axisline"]["color"] = "b"
        axis._axinfo["grid"]["linewidth"] = 0.5
        axis._axinfo["grid"]["linestyle"] = "--"
        axis._axinfo["grid"]["color"] = "#d1d1d1"
        axis._axinfo["tick"]["inward_factor"] = 0.0
        axis._axinfo["tick"]["outward_factor"] = 0.0
        axis.set_pane_color((0, 0, 0))


def vis2(points, weights):
    smoothed = smooth(xy, points, weights,
                      0.005 * np.eye(2)).reshape(n_steps, n_steps)
    smoothed += np.random.default_rng().random(size=smoothed.shape) * 0.005
    smoothed -= smoothed.min()
    smoothed /= smoothed.max()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 10))
    ax.plot_surface(
        xx, yy, smoothed, linewidth=0, antialiased=False, cmap=plt.cm.coolwarm
    )
    ax.set_xlim([-v_max, v_max])
    ax.set_ylim([-v_max, v_max])
    ax.set_zticklabels([])
    ax.set_proj_type("ortho")
    # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.set_axis_off()
    # ax.xaxis._axinfo["grid"]['color'] = (1,1,1,0)
    # ax.yaxis._axinfo["grid"]['color'] = (1,1,1,0)
    # ax.zaxis._axinfo["grid"]['color'] = (1,1,1,0)
    plt.tight_layout()
    plt.savefig("true_small.pdf")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.contourf(xx, yy, smoothed, cmap=plt.cm.coolwarm)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig("true_small_2d.pdf")


def vis0(points, weights):
    smoothed = smooth(xy, points, weights,
                      0.005 * np.eye(2)).reshape(n_steps, n_steps)
    smoothed -= smoothed.min()
    smoothed /= smoothed.max()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 10))
    ax.plot_surface(
        xx, yy, smoothed, linewidth=0, antialiased=False, cmap=plt.cm.coolwarm
    )
    # ax.set_box_aspect(aspect=(2, 1, 1))
    ax.set_xlim([-v_max, v_max])
    ax.set_ylim([-v_max, v_max])
    ax.set_zticklabels([])
    ax.set_proj_type("ortho")
    # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.set_axis_off()
    # ax.xaxis._axinfo["grid"]['color'] = (1,1,1,0)
    # ax.yaxis._axinfo["grid"]['color'] = (1,1,1,0)
    # ax.zaxis._axinfo["grid"]['color'] = (1,1,1,0)
    plt.tight_layout()
    plt.savefig("true.pdf")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.contourf(xx, yy, smoothed, cmap=plt.cm.coolwarm)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig("true_2d.pdf")


def vis3(points, weights):
    smoothed = smooth(xy, points, weights,
                      0.005 * np.eye(2)).reshape(n_steps, n_steps)
    smoothed += np.random.default_rng().random(size=smoothed.shape) * 0.005
    smoothed[smoothed > 0.005] = np.nan
    smoothed -= np.nanmin(smoothed)
    smoothed /= np.nanmax(smoothed)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 10))
    ax.plot_surface(
        xx, yy, smoothed, linewidth=0, antialiased=False, cmap=plt.cm.coolwarm
    )
    # ax.set_box_aspect(aspect=(2, 1, 1))
    ax.set_xlim([-v_max, v_max])
    ax.set_ylim([-v_max, v_max])
    ax.set_zticklabels([])
    ax.set_proj_type("ortho")
    # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.set_axis_off()
    # ax.xaxis._axinfo["grid"]['color'] = (1,1,1,0)
    # ax.yaxis._axinfo["grid"]['color'] = (1,1,1,0)
    # ax.zaxis._axinfo["grid"]['color'] = (1,1,1,0)
    plt.tight_layout()
    plt.savefig("true_small_clip.pdf")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.contourf(xx, yy, smoothed, cmap=plt.cm.coolwarm)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig("true_small_2d_clip.pdf")


def vis4(points, weights):
    out = cv2.VideoWriter(
        "diffusion.avi", cv2.VideoWriter_fourcc(*"xvid"), 20, (1000, 1000)
    )
    out_2d = cv2.VideoWriter(
        "diffusion_2d.avi", cv2.VideoWriter_fourcc(*"xvid"), 20, (1000, 1000)
    )
    smoothed = smooth(xy, points, weights,
                      0.005 * np.eye(2)).reshape(n_steps, n_steps)
    smoothed -= smoothed.min()
    smoothed /= smoothed.max()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 10))
    surf = ax.plot_surface(
        xx, yy, smoothed, linewidth=0, antialiased=False, cmap=plt.cm.coolwarm
    )
    ax.set_xlim([-v_max, v_max])
    ax.set_ylim([-v_max, v_max])
    ax.set_zticklabels([])
    ax.set_proj_type("ortho")
    plt.tight_layout()
    # smoothed += np.random.default_rng().random(size=smoothed.shape) * 0.005
    fig_2d, ax_2d = plt.subplots(figsize=(10, 10))
    text = ax_2d.text(0.5, 0.5, "\\( t = 0 \\)")
    cont_2d = ax_2d.contourf(xx, yy, smoothed, cmap=plt.cm.coolwarm)
    ax_2d.set_axis_off()
    plt.tight_layout()
    for t in ts:
        smoothed = smooth(xy, points, weights, (0.005 + 2 * t) *
                          np.eye(2)).reshape(n_steps, n_steps)
        smoothed -= smoothed.min()
        smoothed /= smoothed.max()
        surf.remove()
        cont_2d.remove()
        cont_2d = ax_2d.contourf(xx, yy, smoothed, cmap=plt.cm.coolwarm)
        surf = ax.plot_surface(
            xx,
            yy,
            smoothed,
            linewidth=0,
            antialiased=False,
            cmap=plt.cm.coolwarm
        )
        text.set_text(f"\\( t = {t:.2f} \\)")
        plt.pause(0.01)
        e_plot = cv2.cvtColor(plot_to_numpy(fig), cv2.COLOR_RGB2BGR)
        e_plot_2d = cv2.cvtColor(plot_to_numpy(fig_2d), cv2.COLOR_RGB2BGR)
        print(smoothed.max())
        out.write(e_plot)
        out_2d.write(e_plot_2d)

    out.release()
    out_2d.release()


def empirical_bayes(points, weights):
    t = 0.005
    smoothed = smooth(xy, points, weights,
                      t * np.eye(2)).reshape(n_steps, n_steps)
    smoothed += np.random.default_rng().random(size=smoothed.shape) * 0.005
    nabla_log_smoothed = -(
        2 * t * nabla_log_smooth(xy_, points, weights,
                                 t * np.eye(2)).reshape(2, n_steps_, n_steps_)
    )
    fig_d, ax_d = plt.subplots(figsize=(10, 10))
    fig_d, ax_d = plt.subplots(figsize=(10, 10))
    # text = ax_d.text(0.5, 0.5, "\\( t = 0 \\)")
    # text.set_bbox(dict(facecolor="white", alpha=0.9, edgecolor="#A30000"))
    cont_d = ax_d.contour(xx, yy, smoothed, cmap="RdBu_r", linewidths=2)
    ax_d.set_xlim([-1.1 * v_max, 1.1 * v_max])
    ax_d.set_ylim([-1.1 * v_max, 1.1 * v_max])
    # cont_nf = ax_d.contour(xx, yy, smoothed, colors='k', linewidths=.5)
    # ax_d.set_axis_off()
    plt.tight_layout()
    quiv = ax_d.quiver(
        xx_,
        yy_,
        nabla_log_smoothed[0],
        nabla_log_smoothed[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        width=.004
    )
    scat = ax_d.scatter(xx_, yy_, color='k')
    # out_d = cv2.VideoWriter(
    #     "diffusion_quiver_diff.avi", cv2.VideoWriter_fourcc(*"xvid"), 20, (1000, 1000)
    # )
    for t in ts:
        smoothed = smooth(xy, points, weights, (0.005 + 2 * t) *
                          np.eye(2)).reshape(n_steps, n_steps)
        nabla_log_smoothed = -(
            2 * t * nabla_log_smooth(
                xy_, points, weights, (0.005 + 2 * t) * np.eye(2)
            ).reshape(2, n_steps_, n_steps_)
        )
        # smoothed -= smoothed.min()
        # smoothed /= smoothed.max()
        cont_d.remove()
        # cont_nf.remove()
        quiv.remove()

        # cont_nf = ax_d.contour(xx, yy, smoothed, colors='k', linewidths=.5)
        cont_d = ax_d.contour(xx, yy, smoothed, cmap="RdBu_r", linewidths=2)
        ax_d.clabel(cont_d, inline=True)

        # text.set_text(f"\\( t = {t:.2f} \\)")
        quiv = ax_d.quiver(
            xx_,
            yy_,
            nabla_log_smoothed[0],
            nabla_log_smoothed[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            width=.004
        )

        sns.despine()
        plt.savefig(f'./out/toy-quiver/{t:.3f}.pdf')
        # plot_diff = cv2.cvtColor(plot_to_numpy(fig_d), cv2.COLOR_RGB2BGR)
    #     out_d.write(plot_diff)

    # out_d.release()


def noise_estimation(points, weights):
    t = 0.005
    smoothed = smooth(xy, points, weights,
                      t * np.eye(2)).reshape(n_steps, n_steps)
    smoothed += np.random.default_rng().random(size=smoothed.shape) * 0.005
    nabla_log_smoothed = -(
        2 * t * nabla_log_smooth(xy, points, weights,
                                 t * np.eye(2)).reshape(2, n_steps, n_steps)
    )
    fig_d, ax_d = plt.subplots(figsize=(10, 10))
    text = ax_d.text(0.5, 0.5, "\\( t = 0 \\)")
    text.set_bbox(dict(facecolor="white", alpha=0.9, edgecolor="#A30000"))
    cont_d = ax_d.contourf(
        xx,
        yy,
        smoothed,
        levels=np.logspace(-6, 1, 20, base=2),
        cmap=plt.cm.coolwarm
    )
    levels = cont_d.levels
    ax_d.set_axis_off()
    plt.tight_layout()

    out_d = cv2.VideoWriter(
        "diffusion_quiver_diff_est.avi",
        cv2.VideoWriter_fourcc(*"MP42"),
        20,
        (1000, 1000),
    )
    out_est = cv2.VideoWriter(
        "est.avi", cv2.VideoWriter_fourcc(*"MP42"), 20, (1000, 1000)
    )
    point_x1 = 23
    point_y1 = 23
    energies1 = np.empty(ts.shape[0])

    point_x2 = 70
    point_y2 = 68
    energies2 = np.empty(ts.shape[0])

    for i, t in enumerate(ts):
        energies1[i] = smooth(
            xy, points, weights, (0.005 + 2 * t) * np.eye(2)
        ).reshape(n_steps, n_steps)[point_y1, point_x1]
        energies2[i] = smooth(
            xy, points, weights, (0.005 + 2 * t) * np.eye(2)
        ).reshape(n_steps, n_steps)[point_y2, point_x2]
    where1 = xy.reshape(n_steps, n_steps, 2)[point_y1, point_x1]
    where2 = xy.reshape(n_steps, n_steps, 2)[point_y2, point_x2]
    scat1 = ax_d.scatter(where1[0], where1[1], c='tab:green')
    scat2 = ax_d.scatter(where2[0], where2[1], c='tab:orange')
    fig_est, ax_est = plt.subplots(figsize=(10, 10))
    (est1, ) = ax_est.plot(ts, energies1, c='tab:green')
    (est2, ) = ax_est.plot(ts, energies2, c='tab:orange')
    plt.tight_layout()
    est1.set_ydata(np.full((ts.shape[0]), np.nan))
    est2.set_ydata(np.full((ts.shape[0]), np.nan))
    for i, t in enumerate(ts):
        smoothed = smooth(xy, points, weights, (0.005 + 2 * t) *
                          np.eye(2)).reshape(n_steps, n_steps)
        cont_d.remove()
        energies_padded1 = np.concatenate(
            (energies1[:i], np.full((ts.shape[0] - i), np.nan))
        )
        energies_padded2 = np.concatenate(
            (energies2[:i], np.full((ts.shape[0] - i), np.nan))
        )
        est1.set_ydata(energies_padded1)
        est2.set_ydata(energies_padded2)
        scat1.remove()
        scat2.remove()

        # est = ax_est.plot(ts, energies[:i])
        cont_d = ax_d.contourf(xx, yy, smoothed, levels, cmap=plt.cm.coolwarm)
        scat1 = ax_d.scatter(where1[0], where1[1], c='tab:green')
        scat2 = ax_d.scatter(where2[0], where2[1], c='tab:orange')
        text.set_text(f"\\( t = {t:.2f} \\)")
        plt.pause(0.01)
        plot_diff = cv2.cvtColor(plot_to_numpy(fig_d), cv2.COLOR_RGB2BGR)
        plot_quiv = cv2.cvtColor(plot_to_numpy(fig_est), cv2.COLOR_RGB2BGR)
        out_d.write(plot_diff)
        out_est.write(plot_quiv)

    out_d.release()
    out_est.release()


# generate_data()
points = np.load("points.npy")
points -= np.mean(points, 0)
weights = np.load("weights.npy")
weights /= weights.sum()
# exit(0)
# points = np.array([[0.0, 0.0]], dtype=np.float64)
# weights = np.array([1.0], dtype=np.float64)
# vis0(points, weights)
# vis1(points, weights)
# vis2(points, weights)
# vis3(points, weights)
# vis4(points, weights)
empirical_bayes(points, weights)
# noise_estimation(points, weights)
