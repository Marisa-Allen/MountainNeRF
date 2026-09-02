import rasterio
import numpy as np
import torch
import torch.nn.functional as F

"""
This is a class that turns a GEOtif DEM into a 
pytourch module that can be quiered with x y 
coordinates in meters and returns the elevation z of that point.
Also in meters
"""
class DEMHeightField(torch.nn.Module):
    def __init__(self, dem_path: str, lon0: float = None, lat0: float = None):
        """
        dem_path - path to the DEM in .tif format, read with rasterio
        lon0, lat0 - center of the pointcloud, computed before or here
        if it is missing
        """
        super().__init__()

        # Opening and reading the DEM using rasterio
        with rasterio.open(dem_path) as src:
            elevation = src.read(1).astype(np.float32)
            transform = src.transform
            nodata = src.nodata
            h, w = elevation.shape

        # getting rows and cols
        rows, cols = np.indices(elevation.shape)
        # changing into arrays
        lons, lats = rasterio.transform.xy(transform, rows, cols)
        lons = np.array(lons, dtype=np.float64)
        lats = np.array(lats, dtype=np.float64)
        # deal with areas where there is not valid elevation data
        valid = np.ones_like(elevation, dtype=bool)
        if nodata is not None:
            valid &= (elevation != nodata)
            
        # if there is no long or lat provided then just take the mean
        # of each
        if lon0 is None or lat0 is None:
            lon0 = lons[valid].mean()
            lat0 = lats[valid].mean()

        # Transformation to ENU cordinates - long and lat into meters
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * np.cos(np.radians(lat0))

        # creating local x, y coordinates
        x0 = (lons - lon0) * m_per_deg_lon
        y0 = (lats - lat0) * m_per_deg_lat

        # areas where there is no data gets filled with average of surroundings
        elevation_filled = np.where(valid, elevation, np.nanmean(elevation[valid]))

        self.x_min, self.x_max = float(x0.min()), float(x0.max())
        self.y_min, self.y_max = float(y0.min()), float(y0.max())
        # saving to pytourch tensor
        self.register_buffer("dem", torch.from_numpy(elevation_filled.astype(np.float32))[None, None])
    
    # Forward step, when x y cordinated are inputted, get the height z
    def forward(self, xy: torch.Tensor):
        # normalisng cooards to grid space
        x = (xy[..., 0] - self.x_min) / (self.x_max - self.x_min) * 2 - 1
        y = (xy[..., 1] - self.y_min) / (self.y_max - self.y_min) * 2 - 1
        # sample grid
        grid = torch.stack([x, -y], dim=-1).view(1, -1, 1, 2).to(self.dem.device)
        # Sample DEM
        z = F.grid_sample(self.dem, grid, align_corners=True, padding_mode="border")
        # Validity mask
        valid = (x.abs() <= 1) & (y.abs() <= 1)
        # return elevation data
        return z.view(-1), valid.view(-1)
