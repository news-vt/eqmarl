import gymnasium


def gymnasium_vector_make(params: dict):
    return gymnasium.vector.make(**params)