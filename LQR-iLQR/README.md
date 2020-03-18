# LQR and iLQR

You will be controlling a simple 2-link planar arm. It is based on the OpenAI gym API with a couple of additions you will need to approximate the dynamics. The environment comes with rendering code so that you can visually see what the arm is doing. The environments themselves deﬁne a cost matrix Q and a cost matrix R. Use these when calculating your trajectories. The rewards returned by the environment are computed using the LQR cost function. The step function includes an additional argument dt. When calculating the ﬁnite differences your dt will be much smaller than the dt that the simulator normally steps at when calling step. So when executing a command you should just use step with an action. When you are trying to approximate the dynamics using ﬁnite diﬀerences you should use step and override the dt argument as well. You can also explicitly set the state of this simulator using the state attribute. You will need this when doing ﬁnite differences. Just set this attribute equal to the q and q˙ values you want before calling step.

## GIF Animation for CartPole
![result](assets/animation.gif)

## Performance

- Action

  ![r](assets/LQR_action.png)
  
- Postion

  ![r](assets/LQR_position.png)
  
- Velocity

  ![r](assets/LQR_velocity.png)
