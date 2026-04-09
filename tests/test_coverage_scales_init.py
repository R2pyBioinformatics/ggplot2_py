"""Extended coverage tests for ggplot2_py.scales.__init__ (scale constructors)."""

import pytest
import numpy as np

from ggplot2_py.scale import (
    Scale,
    ScaleBinned,
    ScaleBinnedPosition,
    ScaleContinuous,
    ScaleContinuousIdentity,
    ScaleContinuousPosition,
    ScaleDiscrete,
    ScaleDiscreteIdentity,
    ScaleDiscretePosition,
    is_scale,
)
from ggplot2_py.scales import (
    # Position continuous
    scale_x_continuous,
    scale_y_continuous,
    scale_x_log10,
    scale_y_log10,
    scale_x_sqrt,
    scale_y_sqrt,
    scale_x_reverse,
    scale_y_reverse,
    # Position discrete
    scale_x_discrete,
    scale_y_discrete,
    # Position binned
    scale_x_binned,
    scale_y_binned,
    # Position date/datetime
    scale_x_date,
    scale_y_date,
    scale_x_datetime,
    scale_y_datetime,
    scale_x_time,
    scale_y_time,
    # Colour/fill continuous
    scale_colour_continuous,
    scale_fill_continuous,
    scale_colour_gradient,
    scale_fill_gradient,
    scale_colour_gradient2,
    scale_fill_gradient2,
    scale_colour_gradientn,
    scale_fill_gradientn,
    # Colour/fill discrete
    scale_colour_discrete,
    scale_fill_discrete,
    scale_colour_hue,
    scale_fill_hue,
    scale_colour_brewer,
    scale_fill_brewer,
    scale_colour_grey,
    scale_fill_grey,
    # Viridis
    scale_colour_viridis_c,
    scale_fill_viridis_c,
    scale_colour_viridis_d,
    scale_fill_viridis_d,
    scale_colour_viridis_b,
    scale_fill_viridis_b,
    # Distiller / fermenter
    scale_colour_distiller,
    scale_fill_distiller,
    scale_colour_fermenter,
    scale_fill_fermenter,
    # Binned / steps
    scale_colour_binned,
    scale_fill_binned,
    scale_colour_steps,
    scale_fill_steps,
    scale_colour_steps2,
    scale_fill_steps2,
    scale_colour_stepsn,
    scale_fill_stepsn,
    # Identity / manual
    scale_colour_identity,
    scale_fill_identity,
    scale_colour_manual,
    scale_fill_manual,
    scale_continuous_identity,
    scale_discrete_identity,
    scale_discrete_manual,
    # Date / datetime / ordinal
    scale_colour_date,
    scale_fill_date,
    scale_colour_datetime,
    scale_fill_datetime,
    scale_colour_ordinal,
    scale_fill_ordinal,
    # Alpha
    scale_alpha,
    scale_alpha_continuous,
    scale_alpha_discrete,
    scale_alpha_binned,
    scale_alpha_identity,
    scale_alpha_manual,
    scale_alpha_ordinal,
    scale_alpha_date,
    scale_alpha_datetime,
    # Size
    scale_size,
    scale_size_continuous,
    scale_size_discrete,
    scale_size_binned,
    scale_size_area,
    scale_size_identity,
    scale_size_manual,
    scale_size_ordinal,
    scale_size_date,
    scale_size_datetime,
    scale_radius,
    # Shape
    scale_shape,
    scale_shape_discrete,
    scale_shape_identity,
    scale_shape_manual,
    scale_shape_ordinal,
    # Linetype
    scale_linetype,
    scale_linetype_discrete,
    scale_linetype_identity,
    scale_linetype_manual,
    # Linewidth
    scale_linewidth,
    scale_linewidth_continuous,
    scale_linewidth_discrete,
    scale_linewidth_binned,
    scale_linewidth_identity,
    scale_linewidth_manual,
    scale_linewidth_ordinal,
    scale_linewidth_date,
    scale_linewidth_datetime,
    # American spelling aliases
    scale_color_continuous,
    scale_color_discrete,
    scale_color_gradient,
    scale_color_gradient2,
    scale_color_gradientn,
    scale_color_hue,
    scale_color_brewer,
    scale_color_distiller,
    scale_color_fermenter,
    scale_color_grey,
    scale_color_viridis_c,
    scale_color_viridis_d,
    scale_color_viridis_b,
    scale_color_binned,
    scale_color_steps,
    scale_color_steps2,
    scale_color_stepsn,
    scale_color_identity,
    scale_color_manual,
    scale_color_date,
    scale_color_datetime,
    scale_color_ordinal,
    # Helpers
    expansion,
)


# ---------------------------------------------------------------------------
# Position continuous scales
# ---------------------------------------------------------------------------

class TestPositionContinuousScales:
    def test_scale_x_continuous(self):
        s = scale_x_continuous()
        assert isinstance(s, ScaleContinuousPosition)
        assert is_scale(s)

    def test_scale_y_continuous(self):
        s = scale_y_continuous()
        assert isinstance(s, ScaleContinuousPosition)

    def test_scale_x_continuous_with_name(self):
        s = scale_x_continuous(name="X Axis")
        assert is_scale(s)

    def test_scale_x_continuous_with_limits(self):
        s = scale_x_continuous(limits=[0, 100])
        assert is_scale(s)

    def test_scale_y_continuous_with_position(self):
        s = scale_y_continuous(position="right")
        assert is_scale(s)

    def test_scale_x_log10(self):
        s = scale_x_log10()
        assert isinstance(s, ScaleContinuousPosition)

    def test_scale_y_log10(self):
        s = scale_y_log10()
        assert isinstance(s, ScaleContinuousPosition)

    def test_scale_x_sqrt(self):
        s = scale_x_sqrt()
        assert isinstance(s, ScaleContinuousPosition)

    def test_scale_y_sqrt(self):
        s = scale_y_sqrt()
        assert isinstance(s, ScaleContinuousPosition)

    def test_scale_x_reverse(self):
        s = scale_x_reverse()
        assert isinstance(s, ScaleContinuousPosition)

    def test_scale_y_reverse(self):
        s = scale_y_reverse()
        assert isinstance(s, ScaleContinuousPosition)


# ---------------------------------------------------------------------------
# Position discrete scales
# ---------------------------------------------------------------------------

class TestPositionDiscreteScales:
    def test_scale_x_discrete(self):
        s = scale_x_discrete()
        assert isinstance(s, ScaleDiscretePosition)

    def test_scale_y_discrete(self):
        s = scale_y_discrete()
        assert isinstance(s, ScaleDiscretePosition)

    def test_scale_x_discrete_with_name(self):
        s = scale_x_discrete(name="Categories")
        assert is_scale(s)

    def test_scale_y_discrete_with_position(self):
        s = scale_y_discrete(position="right")
        assert is_scale(s)


# ---------------------------------------------------------------------------
# Position binned scales
# ---------------------------------------------------------------------------

class TestPositionBinnedScales:
    def test_scale_x_binned(self):
        s = scale_x_binned()
        assert isinstance(s, ScaleBinnedPosition)

    def test_scale_y_binned(self):
        s = scale_y_binned()
        assert isinstance(s, ScaleBinnedPosition)

    def test_scale_x_binned_with_n_breaks(self):
        s = scale_x_binned(n_breaks=5)
        assert is_scale(s)


# ---------------------------------------------------------------------------
# Position date/datetime scales
# ---------------------------------------------------------------------------

class TestPositionDateScales:
    def test_scale_x_date(self):
        s = scale_x_date()
        assert isinstance(s, ScaleContinuousPosition)

    def test_scale_y_date(self):
        s = scale_y_date()
        assert isinstance(s, ScaleContinuousPosition)

    def test_scale_x_datetime(self):
        s = scale_x_datetime()
        assert isinstance(s, ScaleContinuousPosition)

    def test_scale_y_datetime(self):
        s = scale_y_datetime()
        assert isinstance(s, ScaleContinuousPosition)

    def test_time_aliases(self):
        assert scale_x_time is scale_x_datetime
        assert scale_y_time is scale_y_datetime


# ---------------------------------------------------------------------------
# Colour/fill continuous (gradient) scales
# ---------------------------------------------------------------------------

class TestColourContinuousScales:
    def test_scale_colour_continuous(self):
        s = scale_colour_continuous()
        assert isinstance(s, ScaleContinuous)

    def test_scale_fill_continuous(self):
        s = scale_fill_continuous()
        assert isinstance(s, ScaleContinuous)

    def test_scale_colour_gradient(self):
        s = scale_colour_gradient()
        assert isinstance(s, ScaleContinuous)

    def test_scale_fill_gradient(self):
        s = scale_fill_gradient()
        assert isinstance(s, ScaleContinuous)

    def test_scale_colour_gradient_custom_colours(self):
        s = scale_colour_gradient(low="white", high="red")
        assert is_scale(s)

    def test_scale_colour_gradient2(self):
        s = scale_colour_gradient2()
        assert isinstance(s, ScaleContinuous)

    def test_scale_fill_gradient2(self):
        s = scale_fill_gradient2()
        assert isinstance(s, ScaleContinuous)

    def test_scale_colour_gradient2_custom(self):
        s = scale_colour_gradient2(low="blue", mid="white", high="red", midpoint=0)
        assert is_scale(s)

    def test_scale_colour_gradientn(self):
        s = scale_colour_gradientn(colours=["red", "green", "blue"])
        assert isinstance(s, ScaleContinuous)

    def test_scale_fill_gradientn(self):
        s = scale_fill_gradientn(colors=["red", "green", "blue"])
        assert isinstance(s, ScaleContinuous)

    def test_scale_colour_gradientn_no_colours(self):
        with pytest.raises(Exception):
            scale_colour_gradientn()

    def test_scale_fill_gradientn_no_colours(self):
        with pytest.raises(Exception):
            scale_fill_gradientn()

    def test_scale_colour_continuous_with_palette(self):
        s = scale_colour_continuous(palette=["red", "blue"])
        assert is_scale(s)


# ---------------------------------------------------------------------------
# Colour/fill discrete scales
# ---------------------------------------------------------------------------

class TestColourDiscreteScales:
    def test_scale_colour_discrete(self):
        s = scale_colour_discrete()
        assert isinstance(s, ScaleDiscrete)

    def test_scale_fill_discrete(self):
        s = scale_fill_discrete()
        assert isinstance(s, ScaleDiscrete)

    def test_scale_colour_hue(self):
        s = scale_colour_hue()
        assert isinstance(s, ScaleDiscrete)

    def test_scale_fill_hue(self):
        s = scale_fill_hue()
        assert isinstance(s, ScaleDiscrete)

    def test_scale_colour_hue_custom(self):
        s = scale_colour_hue(h=(0, 360), c=50, l=70)
        assert is_scale(s)

    def test_scale_colour_brewer(self):
        s = scale_colour_brewer()
        assert isinstance(s, ScaleDiscrete)

    def test_scale_fill_brewer(self):
        s = scale_fill_brewer()
        assert isinstance(s, ScaleDiscrete)

    def test_scale_colour_brewer_type(self):
        s = scale_colour_brewer(type="qual", palette="Set1")
        assert is_scale(s)

    def test_scale_colour_grey(self):
        s = scale_colour_grey()
        assert isinstance(s, ScaleDiscrete)

    def test_scale_fill_grey(self):
        s = scale_fill_grey()
        assert isinstance(s, ScaleDiscrete)

    def test_scale_colour_grey_custom(self):
        s = scale_colour_grey(start=0.0, end=1.0)
        assert is_scale(s)


# ---------------------------------------------------------------------------
# Viridis scales
# ---------------------------------------------------------------------------

class TestViridisScales:
    def test_scale_colour_viridis_d(self):
        s = scale_colour_viridis_d()
        assert isinstance(s, ScaleDiscrete)

    def test_scale_fill_viridis_d(self):
        s = scale_fill_viridis_d()
        assert isinstance(s, ScaleDiscrete)

    def test_scale_colour_viridis_c(self):
        s = scale_colour_viridis_c()
        assert isinstance(s, ScaleContinuous)

    def test_scale_fill_viridis_c(self):
        s = scale_fill_viridis_c()
        assert isinstance(s, ScaleContinuous)

    def test_scale_colour_viridis_b(self):
        s = scale_colour_viridis_b()
        assert isinstance(s, ScaleBinned)

    def test_scale_fill_viridis_b(self):
        s = scale_fill_viridis_b()
        assert isinstance(s, ScaleBinned)

    def test_scale_colour_viridis_d_custom(self):
        s = scale_colour_viridis_d(option="A", begin=0.2, end=0.8)
        assert is_scale(s)


# ---------------------------------------------------------------------------
# Distiller / fermenter scales
# ---------------------------------------------------------------------------

class TestDistillerFermenterScales:
    def test_scale_colour_distiller(self):
        s = scale_colour_distiller()
        assert isinstance(s, ScaleContinuous)

    def test_scale_fill_distiller(self):
        s = scale_fill_distiller()
        assert isinstance(s, ScaleContinuous)

    def test_scale_colour_fermenter(self):
        s = scale_colour_fermenter()
        assert isinstance(s, ScaleBinned)

    def test_scale_fill_fermenter(self):
        s = scale_fill_fermenter()
        assert isinstance(s, ScaleBinned)


# ---------------------------------------------------------------------------
# Binned / steps scales
# ---------------------------------------------------------------------------

class TestBinnedStepsScales:
    def test_scale_colour_binned(self):
        s = scale_colour_binned()
        assert isinstance(s, ScaleBinned)

    def test_scale_fill_binned(self):
        s = scale_fill_binned()
        assert isinstance(s, ScaleBinned)

    def test_scale_colour_steps(self):
        s = scale_colour_steps()
        assert isinstance(s, ScaleBinned)

    def test_scale_fill_steps(self):
        s = scale_fill_steps()
        assert isinstance(s, ScaleBinned)

    def test_scale_colour_steps2(self):
        s = scale_colour_steps2()
        assert isinstance(s, ScaleBinned)

    def test_scale_fill_steps2(self):
        s = scale_fill_steps2()
        assert isinstance(s, ScaleBinned)

    def test_scale_colour_stepsn(self):
        s = scale_colour_stepsn(colours=["red", "blue", "green"])
        assert isinstance(s, ScaleBinned)

    def test_scale_fill_stepsn(self):
        s = scale_fill_stepsn(colors=["red", "blue", "green"])
        assert isinstance(s, ScaleBinned)

    def test_scale_colour_stepsn_no_colours(self):
        with pytest.raises(Exception):
            scale_colour_stepsn()

    def test_scale_fill_stepsn_no_colours(self):
        with pytest.raises(Exception):
            scale_fill_stepsn()


# ---------------------------------------------------------------------------
# Identity scales
# ---------------------------------------------------------------------------

class TestIdentityScales:
    def test_scale_colour_identity(self):
        s = scale_colour_identity()
        assert isinstance(s, ScaleDiscreteIdentity)

    def test_scale_fill_identity(self):
        s = scale_fill_identity()
        assert isinstance(s, ScaleDiscreteIdentity)

    def test_scale_continuous_identity(self):
        s = scale_continuous_identity("size")
        assert isinstance(s, ScaleContinuousIdentity)

    def test_scale_discrete_identity(self):
        s = scale_discrete_identity("shape")
        assert isinstance(s, ScaleDiscreteIdentity)


# ---------------------------------------------------------------------------
# Manual scales
# ---------------------------------------------------------------------------

class TestManualScales:
    def test_scale_colour_manual_dict(self):
        s = scale_colour_manual(values={"a": "red", "b": "blue"})
        assert isinstance(s, ScaleDiscrete)

    def test_scale_fill_manual_dict(self):
        s = scale_fill_manual(values={"a": "red", "b": "blue"})
        assert isinstance(s, ScaleDiscrete)

    def test_scale_colour_manual_list(self):
        s = scale_colour_manual(values=["red", "blue", "green"])
        assert isinstance(s, ScaleDiscrete)

    def test_scale_discrete_manual(self):
        s = scale_discrete_manual("colour", values={"a": "red"})
        assert isinstance(s, ScaleDiscrete)

    def test_scale_colour_manual_with_breaks(self):
        s = scale_colour_manual(values=["red", "blue"], breaks=["a", "b"])
        assert is_scale(s)


# ---------------------------------------------------------------------------
# Date / datetime / ordinal colour scales
# ---------------------------------------------------------------------------

class TestColourDateOrdinalScales:
    def test_scale_colour_date(self):
        s = scale_colour_date()
        assert isinstance(s, ScaleContinuous)

    def test_scale_fill_date(self):
        s = scale_fill_date()
        assert isinstance(s, ScaleContinuous)

    def test_scale_colour_datetime(self):
        s = scale_colour_datetime()
        assert isinstance(s, ScaleContinuous)

    def test_scale_fill_datetime(self):
        s = scale_fill_datetime()
        assert isinstance(s, ScaleContinuous)

    def test_scale_colour_ordinal(self):
        s = scale_colour_ordinal()
        assert isinstance(s, ScaleDiscrete)

    def test_scale_fill_ordinal(self):
        s = scale_fill_ordinal()
        assert isinstance(s, ScaleDiscrete)


# ---------------------------------------------------------------------------
# Alpha scales
# ---------------------------------------------------------------------------

class TestAlphaScales:
    def test_scale_alpha(self):
        s = scale_alpha()
        assert isinstance(s, ScaleContinuous)

    def test_scale_alpha_continuous_alias(self):
        assert scale_alpha_continuous is scale_alpha

    def test_scale_alpha_with_range(self):
        s = scale_alpha(range=(0.2, 0.8))
        assert is_scale(s)

    def test_scale_alpha_binned(self):
        s = scale_alpha_binned()
        assert isinstance(s, ScaleBinned)

    def test_scale_alpha_binned_with_range(self):
        s = scale_alpha_binned(range=(0.1, 1.0))
        assert is_scale(s)

    def test_scale_alpha_discrete(self):
        s = scale_alpha_discrete()
        assert isinstance(s, ScaleDiscrete)

    def test_scale_alpha_ordinal(self):
        s = scale_alpha_ordinal()
        assert isinstance(s, ScaleDiscrete)

    def test_scale_alpha_ordinal_with_range(self):
        s = scale_alpha_ordinal(range=(0.2, 0.9))
        assert is_scale(s)

    def test_scale_alpha_identity(self):
        s = scale_alpha_identity()
        assert isinstance(s, ScaleContinuousIdentity)

    def test_scale_alpha_manual(self):
        s = scale_alpha_manual(values={"a": 0.5, "b": 1.0})
        assert isinstance(s, ScaleDiscrete)

    def test_scale_alpha_date(self):
        s = scale_alpha_date()
        assert isinstance(s, ScaleContinuous)

    def test_scale_alpha_date_with_range(self):
        s = scale_alpha_date(range=(0.2, 0.9))
        assert is_scale(s)

    def test_scale_alpha_datetime(self):
        s = scale_alpha_datetime()
        assert isinstance(s, ScaleContinuous)


# ---------------------------------------------------------------------------
# Size scales
# ---------------------------------------------------------------------------

class TestSizeScales:
    def test_scale_size(self):
        s = scale_size()
        assert isinstance(s, ScaleContinuous)

    def test_scale_size_continuous_alias(self):
        assert scale_size is scale_size_continuous

    def test_scale_size_with_range(self):
        s = scale_size_continuous(range=(1, 10))
        assert is_scale(s)

    def test_scale_size_discrete(self):
        s = scale_size_discrete()
        assert isinstance(s, ScaleDiscrete)

    def test_scale_size_binned(self):
        s = scale_size_binned()
        assert isinstance(s, ScaleBinned)

    def test_scale_size_area(self):
        s = scale_size_area()
        assert isinstance(s, ScaleContinuous)

    def test_scale_size_identity(self):
        s = scale_size_identity()
        assert isinstance(s, ScaleContinuousIdentity)

    def test_scale_size_manual(self):
        s = scale_size_manual(values={"a": 1, "b": 5})
        assert isinstance(s, ScaleDiscrete)

    def test_scale_size_ordinal(self):
        s = scale_size_ordinal()
        assert isinstance(s, ScaleDiscrete)

    def test_scale_size_date(self):
        s = scale_size_date()
        assert isinstance(s, ScaleContinuous)

    def test_scale_size_datetime(self):
        s = scale_size_datetime()
        assert isinstance(s, ScaleContinuous)

    def test_scale_radius(self):
        s = scale_radius()
        assert isinstance(s, ScaleContinuous)


# ---------------------------------------------------------------------------
# Shape scales
# ---------------------------------------------------------------------------

class TestShapeScales:
    def test_scale_shape(self):
        s = scale_shape()
        assert isinstance(s, ScaleDiscrete)

    def test_scale_shape_discrete_alias(self):
        assert scale_shape is scale_shape_discrete

    def test_scale_shape_identity(self):
        s = scale_shape_identity()
        assert isinstance(s, (ScaleDiscreteIdentity, ScaleContinuousIdentity))

    def test_scale_shape_manual(self):
        s = scale_shape_manual(values={"a": 16, "b": 17})
        assert isinstance(s, ScaleDiscrete)

    def test_scale_shape_ordinal(self):
        s = scale_shape_ordinal()
        assert isinstance(s, ScaleDiscrete)


# ---------------------------------------------------------------------------
# Linetype scales
# ---------------------------------------------------------------------------

class TestLinetypeScales:
    def test_scale_linetype(self):
        s = scale_linetype()
        assert isinstance(s, ScaleDiscrete)

    def test_scale_linetype_discrete_alias(self):
        assert scale_linetype is scale_linetype_discrete

    def test_scale_linetype_identity(self):
        s = scale_linetype_identity()
        assert isinstance(s, ScaleDiscreteIdentity)

    def test_scale_linetype_manual(self):
        s = scale_linetype_manual(values={"a": "solid", "b": "dashed"})
        assert isinstance(s, ScaleDiscrete)


# ---------------------------------------------------------------------------
# Linewidth scales
# ---------------------------------------------------------------------------

class TestLinewidthScales:
    def test_scale_linewidth(self):
        s = scale_linewidth()
        assert isinstance(s, ScaleContinuous)

    def test_scale_linewidth_continuous_alias(self):
        assert scale_linewidth is scale_linewidth_continuous

    def test_scale_linewidth_with_range(self):
        s = scale_linewidth(range=(1, 5))
        assert is_scale(s)

    def test_scale_linewidth_discrete(self):
        s = scale_linewidth_discrete()
        assert isinstance(s, ScaleDiscrete)

    def test_scale_linewidth_binned(self):
        s = scale_linewidth_binned()
        assert isinstance(s, ScaleBinned)

    def test_scale_linewidth_identity(self):
        s = scale_linewidth_identity()
        assert isinstance(s, ScaleContinuousIdentity)

    def test_scale_linewidth_manual(self):
        s = scale_linewidth_manual(values={"a": 1, "b": 3})
        assert isinstance(s, ScaleDiscrete)

    def test_scale_linewidth_ordinal(self):
        s = scale_linewidth_ordinal()
        assert isinstance(s, ScaleDiscrete)

    def test_scale_linewidth_ordinal_with_range(self):
        s = scale_linewidth_ordinal(range=(1, 5))
        assert is_scale(s)

    def test_scale_linewidth_date(self):
        s = scale_linewidth_date()
        assert isinstance(s, ScaleContinuous)

    def test_scale_linewidth_datetime(self):
        s = scale_linewidth_datetime()
        assert isinstance(s, ScaleContinuous)


# ---------------------------------------------------------------------------
# American spelling aliases
# ---------------------------------------------------------------------------

class TestAmericanSpellingAliases:
    def test_color_continuous(self):
        assert scale_color_continuous is scale_colour_continuous

    def test_color_discrete(self):
        assert scale_color_discrete is scale_colour_discrete

    def test_color_gradient(self):
        assert scale_color_gradient is scale_colour_gradient

    def test_color_gradient2(self):
        assert scale_color_gradient2 is scale_colour_gradient2

    def test_color_gradientn(self):
        assert scale_color_gradientn is scale_colour_gradientn

    def test_color_hue(self):
        assert scale_color_hue is scale_colour_hue

    def test_color_brewer(self):
        assert scale_color_brewer is scale_colour_brewer

    def test_color_distiller(self):
        assert scale_color_distiller is scale_colour_distiller

    def test_color_fermenter(self):
        assert scale_color_fermenter is scale_colour_fermenter

    def test_color_grey(self):
        assert scale_color_grey is scale_colour_grey

    def test_color_viridis_c(self):
        assert scale_color_viridis_c is scale_colour_viridis_c

    def test_color_viridis_d(self):
        assert scale_color_viridis_d is scale_colour_viridis_d

    def test_color_viridis_b(self):
        assert scale_color_viridis_b is scale_colour_viridis_b

    def test_color_binned(self):
        assert scale_color_binned is scale_colour_binned

    def test_color_steps(self):
        assert scale_color_steps is scale_colour_steps

    def test_color_steps2(self):
        assert scale_color_steps2 is scale_colour_steps2

    def test_color_stepsn(self):
        assert scale_color_stepsn is scale_colour_stepsn

    def test_color_identity(self):
        assert scale_color_identity is scale_colour_identity

    def test_color_manual(self):
        assert scale_color_manual is scale_colour_manual

    def test_color_date(self):
        assert scale_color_date is scale_colour_date

    def test_color_datetime(self):
        assert scale_color_datetime is scale_colour_datetime

    def test_color_ordinal(self):
        assert scale_color_ordinal is scale_colour_ordinal


# ---------------------------------------------------------------------------
# expansion helper
# ---------------------------------------------------------------------------

class TestExpansionExtended:
    def test_mult_only(self):
        e = expansion(mult=0.05)
        assert len(e) == 4

    def test_add_only(self):
        e = expansion(add=1)
        assert len(e) == 4

    def test_both(self):
        e = expansion(mult=0.05, add=1)
        assert len(e) == 4
        assert float(e[0]) == pytest.approx(0.05)
        assert float(e[1]) == pytest.approx(1.0)

    def test_asymmetric(self):
        e = expansion(mult=(0.05, 0.1))
        assert len(e) == 4
