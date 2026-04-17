from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


SortField = Literal[
    "patient_id",
    "age",
    "gender",
    "ecog_score",
    "tumor_location",
    "histology_type",
    "clinical_stage",
    "cea_level",
    "mmr_status",
]
SortDirection = Literal["asc", "desc"]


class DatabaseFilters(BaseModel):
    patient_id: int | None = None
    tumor_location: list[str] = Field(default_factory=list)
    ct_stage: list[str] = Field(default_factory=list)
    cn_stage: list[str] = Field(default_factory=list)
    histology_type: list[str] = Field(default_factory=list)
    mmr_status: list[str] = Field(default_factory=list)
    age_min: int | None = None
    age_max: int | None = None
    cea_max: float | None = None
    family_history: bool | None = None
    biopsy_confirmed: bool | None = None
    ecog_min: int | None = Field(default=None, ge=0, le=5)
    ecog_max: int | None = Field(default=None, ge=0, le=5)

    model_config = ConfigDict(extra="forbid")


class DatabasePagination(BaseModel):
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)

    model_config = ConfigDict(extra="forbid")


class DatabaseSort(BaseModel):
    field: SortField = "patient_id"
    direction: SortDirection = "asc"

    model_config = ConfigDict(extra="forbid")


class DatabaseSearchRequest(BaseModel):
    filters: DatabaseFilters = Field(default_factory=DatabaseFilters)
    pagination: DatabasePagination = Field(default_factory=DatabasePagination)
    sort: DatabaseSort = Field(default_factory=DatabaseSort)

    model_config = ConfigDict(extra="forbid")


class DatabaseSearchResponse(BaseModel):
    items: list[dict[str, Any]]
    total: int
    page: int
    page_size: int
    applied_filters: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class DatabaseNumericStatistics(BaseModel):
    min: float | int | None = None
    max: float | int | None = None
    mean: float | None = None

    model_config = ConfigDict(extra="forbid")


class DatabaseStatsResponse(BaseModel):
    total_cases: int = 0
    gender_distribution: dict[str, int] = Field(default_factory=dict)
    age_statistics: DatabaseNumericStatistics = Field(default_factory=DatabaseNumericStatistics)
    tumor_location_distribution: dict[str, int] = Field(default_factory=dict)
    ct_stage_distribution: dict[str, int] = Field(default_factory=dict)
    mmr_status_distribution: dict[str, int] = Field(default_factory=dict)
    cea_statistics: DatabaseNumericStatistics = Field(default_factory=DatabaseNumericStatistics)

    model_config = ConfigDict(extra="forbid")


class DatabaseAvailableData(BaseModel):
    case_info: bool
    imaging: bool
    pathology_slides: bool

    model_config = ConfigDict(extra="forbid")


class DatabaseCaseDetailResponse(BaseModel):
    patient_id: str
    case_record: dict[str, Any] | None = None
    available_data: DatabaseAvailableData
    cards: dict[str, dict[str, Any]] = Field(default_factory=dict)


class DatabaseUpsertRequest(BaseModel):
    record: dict[str, Any]

    model_config = ConfigDict(extra="forbid")


class DatabaseQueryIntentRequest(BaseModel):
    query: str = Field(min_length=1)

    model_config = ConfigDict(extra="forbid")


class DatabaseQueryIntentResponse(BaseModel):
    query: str
    normalized_query: str
    filters: dict[str, Any] = Field(default_factory=dict)
    unsupported_terms: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)