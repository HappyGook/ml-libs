from manim import *
from typing_extensions import runtime


class GradientScene(ThreeDScene):

    def __init__(self, func, classic_path, momentum_path, **kwargs):
        super().__init__(**kwargs)

        self.func = func
        self.classic_path = classic_path
        self.momentum_path = momentum_path


    def construct(self):

        # Camera orientation
        self.set_camera_orientation(
            phi=65 * DEGREES,
            theta=45 * DEGREES
        )

        # Axes
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[0, 10, 2],
        )

        self.add(axes)

        # Choose surface
        surface = self.create_surface(axes, self.func)

        surface.set_style(
            fill_opacity=0.6,
            stroke_color=GRAY
        )

        self.add(surface)

        # Create balls
        ball_gd = Sphere(radius=0.07, color=BLUE)
        ball_momentum = Sphere(radius=0.07, color=RED)

        # Initial positions
        x0, y0 = self.classic_path[0]

        ball_gd.move_to(
            axes.c2p(x0, y0, self.func([x0, y0]))
        )

        x1, y1 = self.momentum_path[0]

        ball_momentum.move_to(
            axes.c2p(x1, y1, self.func([x1, y1]))
        )

        self.add(ball_gd, ball_momentum)

        # Animate trajectories
        self.animate_paths(axes, ball_gd, ball_momentum)

    # Build surface based on chosen surface func
    def create_surface(self, axes, func):

        surface = Surface(
            lambda u, v: axes.c2p(
                u,
                v,
                func([u, v])
            ),
            u_range=[-2, 2],
            v_range=[-2, 2],
            resolution=(40, 40)
        )

        return surface


    def bowl_surface(self, axes):

        surface = Surface(
            lambda u, v: axes.c2p(
                u,
                v,
                u**2 + v**2
            ),
            u_range=[-2, 2],
            v_range=[-2, 2],
            resolution=(40, 40)
        )

        return surface


    def valley_surface(self, axes):

        surface = Surface(
            lambda u, v: axes.c2p(
                u,
                v,
                u**2 + 10 * v**2
            ),
            u_range=[-2, 2],
            v_range=[-2, 2],
            resolution=(40, 40)
        )

        return surface


    def rosenbrock_surface(self, axes):

        surface = Surface(
            lambda u, v: axes.c2p(
                u,
                v,
                (1 - u) ** 2 + 100 * (v - u ** 2) ** 2
            ),
            u_range=[-2, 2],
            v_range=[-1, 3],
            resolution=(40, 40)
        )

        return surface

    # Animate the paths of balls
    def animate_paths(self, axes, ball_gd, ball_momentum):
        for p1, p2 in zip(self.classic_path, self.momentum_path):
            x1, y1 = p1
            x2, y2 = p2

            z1 = self.func([x1, y1])
            z2 = self.func([x2, y2])

            self.play(
                ball_gd.animate.move_to(
                    axes.c2p(x1,y1,z1)
                ),
                ball_momentum.animate.move_to(
                    axes.c2p(x2,y2,z2)
                ),
                run_time=0.1
            )