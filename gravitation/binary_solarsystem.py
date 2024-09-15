from solarsystem import SolarSystem, Star, Planet

solar_system = SolarSystem(width=1400, height = 900)

stars = (
    Star(solar_system, mass=10_000, position=(-200, 0), velocity=(0, 3.5)),
    Star(solar_system, mass=9000, position=(200, 0), velocity=(0, -3.5)),
    
    Star(solar_system, mass=500, position=(0, 0), velocity=(0,1 )),
)

planets = (
    Planet(solar_system, mass=20, position=(50, 0), velocity=(0, 11)),
    Planet(solar_system, mass=3, position=(-350, 0), velocity=(0, -10)),
    Planet(solar_system, mass=1, position=(0, 200), velocity=(-2, -7)),
)


interval = int(1000 / 60)

def animate():
    solar_system.calculate_all_body_interactions()
    solar_system.update_all()
    solar_system.solar_system.ontimer(animate, interval)

animate()

solar_system.solar_system.mainloop()()
