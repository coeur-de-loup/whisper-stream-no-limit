from fastapi.responses import JSONResponse
from fastapi import APIRouter

router = APIRouter()


# Catch-all route handler for preflight requests and documentation
@router.options("/{full_path:path}")
async def options_route(full_path: str):
    return JSONResponse(content={"message": "OK"})
