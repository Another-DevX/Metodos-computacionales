from solarsystem import SolarSystem, Star, Planet

solar_system = SolarSystem(width = 1400, height = 900)
star = Star(solar_system, mass=10_000)
planet = Planet(solar_system,mass = 1, position=(-350, 0), velocity=(0,1))
planet = Planet(solar_system,mass = 2, position=(-270, 0), velocity=(0,7))


interval = int(1000 / 60)

def animate():
    solar_system.calculate_all_body_interactions()
    solar_system.update_all()
    solar_system.solar_system.ontimer(animate, interval)

animate()

solar_system.solar_system.mainloop()
