// The skin of the head the electrodes are drawn on: one matte colour, no light and no shine, a
// hint of the face from above the viewer, an outline where the skin turns away, and the neck
// fading out below the chin (HeadScene). The colour is the material's tint; how much stronger
// or weaker the outline is than the skin, the first of its custom values.
#include <metal_stdlib>
#include <RealityKit/RealityKit.h>
using namespace metal;

constant float outlineSharpness = 2.4;  // the higher, the thinner the outline
constant float formDepth = 0.14;  // how much the hint of the face darkens the skin turned away from it
constant float3 formFrom = float3(-0.3, 0.6, 0.75);  // where it comes from: above the viewer, a little left
constant float neckGone = -0.128;  // m below the middle of the head: the neck has faded away
constant float neckWhole = -0.078;  // and here it is whole: just under the chin

[[visible]]
void headSurface(realitykit::surface_parameters params)
{
    float3 normal = normalize(params.geometry().normal());
    float3 toViewer = normalize(params.geometry().view_direction());
    float outline = pow(1.0 - abs(dot(normal, toViewer)), outlineSharpness);
    float form = 1.0 - formDepth + formDepth * dot(normal, normalize(formFrom));
    half3 skin = half3(params.material_constants().base_color_tint());
    half3 rim = skin * half(params.uniforms().custom_parameter().x);
    half3 colour = mix(skin * half(form), rim, half(outline));
    params.surface().set_base_color(colour);
    params.surface().set_emissive_color(colour);  // what an unlit material shows
    params.surface().set_opacity(half(smoothstep(neckGone, neckWhole, params.geometry().model_position().y)));
}
