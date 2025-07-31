import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import math

# Constants
DEFAULT_WIDTH, DEFAULT_HEIGHT = 800, 600
FPS = 60
GRAVITY = 0.05
THRUST_POWER = 0.2
LATERAL_POWER = 0.1
DAMPING = 0.995
NUM_HOOPS = 5
BASE_HOOP_SPEED = 1.0
SPEED_INCREMENT = 0.5
MIN_HOOP_SPACING = 100
SEPARATION_FORCE = 0.05

WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
DARK_GRAY = (50, 50, 50)
PURPLE = (200, 0, 200)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GLOW_COLOR = (255, 255, 200)

class Particle:
    def __init__(self, pos, vel, lifetime, color, size=4):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(vel)
        self.lifetime = lifetime
        self.color = color
        self.size = size

    def update(self):
        self.pos += self.vel
        self.lifetime -= 1
        self.vel *= 0.95

    def draw(self, screen):
        if self.lifetime > 0:
            alpha = max(0, min(255, int(255 * (self.lifetime / 30))))
            base_color = tuple(int(min(255, max(0, c))) for c in self.color[:3])
            rgba_color = (*base_color, alpha)
            surf = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
            surf.fill(rgba_color)
            screen.blit(surf, self.pos)

class Platform:
    def __init__(self, x, y, width=120, height=10):
        self.rect = pygame.Rect(x - width // 2, y - height, width, height)
        self.poles = [pygame.Rect(self.rect.left + 10, y - height, 5, 30),
                      pygame.Rect(self.rect.right - 15, y - height, 5, 30)]

    def draw(self, screen):
        pygame.draw.rect(screen, DARK_GRAY, self.rect)
        pygame.draw.rect(screen, GRAY, self.rect.inflate(-10, -4))
        for p in self.poles:
            pygame.draw.rect(screen, DARK_GRAY, p)

    def land(self, rocket):
        rocket.pos = pygame.math.Vector2(self.rect.centerx, self.rect.top - rocket.rect.height / 2)
        rocket.rect.center = rocket.pos
        rocket.vel = pygame.math.Vector2(0, 0)

class Hoop:
    def __init__(self, x, y, speed, radius=40, death_trap=False):
        self.pos = pygame.math.Vector2(x, y)
        angle = random.uniform(0, 360)
        self.speed = speed
        self.vel = pygame.math.Vector2(math.cos(math.radians(angle)), math.sin(math.radians(angle))) * speed
        self.radius = radius
        self.passed = False
        self.death_trap = death_trap

    def update(self, width, height, others):
        self.pos += self.vel
        if self.pos.x - self.radius < 0 or self.pos.x + self.radius > width:
            self.vel.x *= -1
        if self.pos.y - self.radius < 0 or self.pos.y + self.radius > height:
            self.vel.y *= -1
        for other in others:
            if other is self: continue
            offset = self.pos - other.pos
            dist = offset.length()
            min_dist = self.radius + other.radius + MIN_HOOP_SPACING / 4
            if dist < min_dist and dist > 0:
                self.vel += offset.normalize() * SEPARATION_FORCE
        if self.vel.length() > self.speed * 2:
            self.vel.scale_to_length(self.speed * 2)

    def draw(self, screen):
        color = PURPLE if self.death_trap else (GREEN if not self.passed else BLUE)
        pygame.draw.circle(screen, color, (int(self.pos.x), int(self.pos.y)), self.radius, 5)

    def check_pass(self, rocket):
        if rocket.pos.distance_to(self.pos) < self.radius - 5:
            if self.death_trap:
                return "death"
            elif not self.passed:
                self.passed = True
                return True
        return False

class Rocket(pygame.sprite.Sprite):
    def __init__(self, platform):
        super().__init__()
        self.image = pygame.Surface((30, 60), pygame.SRCALPHA)
        pygame.draw.polygon(self.image, WHITE, [(15, 0), (0, 60), (30, 60)])
        self.glow = pygame.Surface((50, 70), pygame.SRCALPHA)
        pygame.draw.ellipse(self.glow, (*GLOW_COLOR, 80), self.glow.get_rect())
        self.pos = pygame.math.Vector2(0, 0)
        self.rect = self.image.get_rect()
        platform.land(self)
        self.vel = pygame.math.Vector2(0, 0)
        self.particles = []

    def update(self, _, width, height, landed):
        if landed: return
        self.vel.y += GRAVITY
        self.vel *= DAMPING
        self.pos += self.vel
        half_w, half_h = self.rect.width / 2, self.rect.height / 2
        self.pos.x = max(half_w, min(self.pos.x, width - half_w))
        self.pos.y = max(half_h, min(self.pos.y, height - half_h))
        self.rect.center = self.pos
        for p in self.particles[:]:
            p.update()
            if p.lifetime <= 0:
                self.particles.remove(p)

    def draw(self, screen):
        screen.blit(self.glow, self.glow.get_rect(center=self.pos))
        for p in self.particles:
            p.draw(screen)
        screen.blit(self.image, self.rect)

class RocketHoopsEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, width=800, height=600, render_mode=None):
        super().__init__()
        self.width = width
        self.height = height
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(5)  # NOP, UP, DOWN, LEFT, RIGHT
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.platform = Platform(self.width // 2, self.height - 20)
        self.rocket = Rocket(self.platform)
        self.hoop_speed = BASE_HOOP_SPEED
        self.hoops = [self._spawn_hoop() for _ in range(NUM_HOOPS)]
        self.score = 0
        self.done = False
        self.screen = None
        self.clock = None
        self.surface = None
        self.episode_start_time = None
        self.last_hoop_time = None

    def _spawn_hoop(self, death=False):
        while True:
            x, y = random.randint(100, self.width - 100), random.randint(100, self.height - 200)
            return Hoop(x, y, self.hoop_speed, death_trap=death)

    def _get_obs(self):
        pos = self.rocket.pos
        vel = self.rocket.vel
        next_hoop = next((h for h in self.hoops if not h.passed and not h.death_trap), None)
        dx, dy = 0, 0
        if next_hoop:
            dx = next_hoop.pos.x - pos.x
            dy = next_hoop.pos.y - pos.y
        return np.array([pos.x, pos.y, vel.x, vel.y, dx, dy], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.platform = Platform(self.width // 2, self.height - 20)
        self.rocket = Rocket(self.platform)
        self.hoop_speed = BASE_HOOP_SPEED
        self.hoops = [self._spawn_hoop() for _ in range(NUM_HOOPS)]
        self.score = 0
        self.done = False
        self.episode_start_time = pygame.time.get_ticks()
        self.last_hoop_time = self.episode_start_time
        return self._get_obs(), {}

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, self.done, False, {}

        if action == 1: self.rocket.vel.y -= THRUST_POWER
        elif action == 2: self.rocket.vel.y += THRUST_POWER
        elif action == 3: self.rocket.vel.x -= LATERAL_POWER
        elif action == 4: self.rocket.vel.x += LATERAL_POWER

        self.rocket.update([False]*300, self.width, self.height, self.done)

        current_time = pygame.time.get_ticks()
        time_since_last = current_time - self.last_hoop_time
        reward = -0.01  # Per-frame penalty to encourage speed

        for h in self.hoops:
            h.update(self.width, self.height, self.hoops)
            result = h.check_pass(self.rocket)
            if result == "death":
                self.done = True
                reward = -10
                break
            elif result:
                # Time-based hoop reward
                hoop_time_reward = max(2.0, 5.0 * (1000 / (time_since_last + 1)))
                reward += hoop_time_reward
                self.score += 1
                self.last_hoop_time = current_time
                self.hoop_speed += SPEED_INCREMENT
                self.hoops.append(self._spawn_hoop(death=True))
                for hh in self.hoops:
                    hh.speed = self.hoop_speed
                    hh.vel.scale_to_length(self.hoop_speed)
                break

        # Completion bonus
        if self.score >= NUM_HOOPS:
            total_time = current_time - self.episode_start_time
            time_bonus = max(100.0, 1000.0 * (5000 / (total_time + 1)))
            reward += time_bonus
            self.done = True

        return self._get_obs(), reward, self.done, False, {}

    def render(self):
        if self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((self.width, self.height))
                self.clock = pygame.time.Clock()
            self._draw(self.screen)
            pygame.display.flip()
            self.clock.tick(FPS)

        elif self.render_mode == "rgb_array":
            if self.surface is None:
                pygame.init()
                self.surface = pygame.Surface((self.width, self.height))
            self._draw(self.surface)
            return np.transpose(
                pygame.surfarray.array3d(self.surface), axes=(1, 0, 2)
            )

    def _draw(self, surface):
        surface.fill((0, 0, 0))
        for h in self.hoops: h.draw(surface)
        self.platform.draw(surface)
        self.rocket.draw(surface)

    def close(self):
        if self.screen or self.surface:
            pygame.quit()