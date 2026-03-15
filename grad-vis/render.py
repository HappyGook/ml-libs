from manim import *


class GradientScene(ThreeDScene):

    def __init__(self, func, classic_path, momentum_path, **kwargs):
        super().__init__(**kwargs)
        self.func = func
        self.classic_path = classic_path
        self.momentum_path = momentum_path

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=45 * DEGREES)

        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[0, 10, 2],
        )
        self.add(axes)

        surface = self.create_surface(axes, self.func)
        surface.set_style(fill_opacity=0.6, stroke_color=GRAY)
        self.add(surface)

        ball_gd = Sphere(radius=0.07)
        ball_gd.set_color(BLUE)

        ball_momentum = Sphere(radius=0.07)
        ball_momentum.set_color(RED)

        x0, y0 = self.classic_path[0]
        ball_gd.move_to(axes.c2p(x0, y0, self.func([x0, y0])))

        x1, y1 = self.momentum_path[0]
        ball_momentum.move_to(axes.c2p(x1, y1, self.func([x1, y1])))

        trail_gd = TracedPath(
            ball_gd.get_center,
            stroke_color=BLUE,
            stroke_width=2,
            stroke_opacity=0.8,
        )
        trail_momentum = TracedPath(
            ball_momentum.get_center,
            stroke_color=RED,
            stroke_width=2,
            stroke_opacity=0.8,
        )

        self.add(trail_gd, trail_momentum)
        self.add(ball_gd, ball_momentum)

        self.animate_paths(axes, ball_gd, ball_momentum)

    def create_surface(self, axes, func):
        return Surface(
            lambda u, v: axes.c2p(u, v, func([u, v])),
            u_range=[-2, 2],
            v_range=[-2, 2],
            resolution=(40, 40),
        )

    def animate_paths(self, axes, ball_gd, ball_momentum):
        for p1, p2 in zip(self.classic_path, self.momentum_path):
            x1, y1 = p1
            x2, y2 = p2

            z1 = self.func([x1, y1])
            z2 = self.func([x2, y2])

            # pan camera toward the midpoint of the two balls
            mid = axes.c2p(
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                (z1 + z2) / 2,
            )

            self.move_camera(
                frame_center=mid,
                run_time=0.2,
                added_anims=[
                    ball_gd.animate.move_to(axes.c2p(x1, y1, z1)),
                    ball_momentum.animate.move_to(axes.c2p(x2, y2, z2)),
                ]
            )