"""
    leaflet.jl

Interactive Leaflet-based HTML visualization engine integration.
All core and specialized Leaflet interactive dashboards and the `LeafletMap` type
are now canonically defined in `TethysPlots` and re-exported here for full
backward compatibility across TethysBayes models and diagnostics.
"""

# Re-exported from TethysPlots via TethysBayes.jl:
# LeafletMap, save_html, save_plot, utm_to_lonlat, lonlat_to_utm,
# leaflet_choropleth, leaflet_spatial_graph, leaflet_hsi_map,
# leaflet_diffusion_map, leaflet_residence_time_map, leaflet_advection_arrows,
# leaflet_spacetime_map, leaflet_current_density_map,
# leaflet_hydrodynamic_dashboard, leaflet_particle_tracks
