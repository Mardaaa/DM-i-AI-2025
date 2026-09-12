import time
import uvicorn
import datetime
from threading import Lock
from fastapi import Body, FastAPI
from dtos import RaceCarPredictRequestDto, RaceCarPredictResponseDto
from expert import ExpertController
from settings import driver_config

HOST = "0.0.0.0"
PORT = 9052


app = FastAPI()
start_time = time.time()
controller = ExpertController(driver_config())
controller_lock = Lock()

@app.post('/predict', response_model=RaceCarPredictResponseDto)
def predict(request: RaceCarPredictRequestDto = Body(...)):
    # The public DTO has no game/session ID: serve one game stream per worker.
    with controller_lock:
        actions = controller.actions(request.model_dump())
    return RaceCarPredictResponseDto(actions=actions)

@app.get('/api')
def hello():
    return {
        "service": "race-car-usecase",
        "uptime": '{}'.format(datetime.timedelta(seconds=time.time() - start_time))
    }


@app.get('/')
def index():
    return "Your endpoint is running!"




if __name__ == '__main__':

    uvicorn.run(
        'api:app',
        host=HOST,
        port=PORT
    )
