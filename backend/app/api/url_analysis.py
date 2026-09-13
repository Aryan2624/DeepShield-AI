from fastapi import APIRouter, HTTPException, status

from app.schemas.url_analysis import (
    URLAnalysisRequest,
    URLAnalysisResponse,
)
from app.services.url_detection_service import analyze_url


router = APIRouter(
    prefix="/api/url",
    tags=["URL Security"],
)


@router.post(
    "/analyze",
    response_model=URLAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze a URL for cybersecurity threats",
)
def analyze_url_endpoint(
    request: URLAnalysisRequest,
):
    try:
        result = analyze_url(
            request.url
        )

        return result

    except (ValueError, TypeError) as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "URL detection model is currently unavailable."
            ),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "An unexpected error occurred while "
                "analyzing the URL."
            ),
        ) from error