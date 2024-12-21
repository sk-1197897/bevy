use bevy_asset::Handle;
use bevy_core_pipeline::fullscreen_vertex_shader::fullscreen_shader_vertex_state;
use bevy_ecs::{
    prelude::{Component, Entity},
    query::QueryItem,
    system::{Commands, Query, Res, ResMut, Resource},
    world::{FromWorld, World},
};
use bevy_math::{AspectRatio, URect};
use bevy_render::{
    camera::Camera,
    extract_component::{ComponentUniforms, ExtractComponent},
    render_resource::{
        binding_types::{sampler, texture_2d, uniform_buffer},
        *,
    },
    renderer::RenderDevice,
    view::ViewTarget,
};
use bevy_utils::tracing::warn;

use super::{
    render::{VolumetricFogTexture, FOG_TEXTURE_FORMAT},
    VolumetricFog,
};

#[derive(Component)]
pub struct UpsamplingBindGroup {
    pub upsampling_bind_group: BindGroup,
    pub sampler: Sampler,
}

#[derive(Component, ShaderType, Clone)]
pub struct UpsamplingUniforms {
    pub aspect: f32,
}

pub fn prepare_upsampling_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    upsampling_pipeline: Res<UpsamplingPipeline>,
    views: Query<(Entity, &VolumetricFogTexture)>,
    uniforms: Res<ComponentUniforms<UpsamplingUniforms>>,
) {
    let sampler = render_device.create_sampler(&SamplerDescriptor {
        min_filter: FilterMode::Linear,
        mag_filter: FilterMode::Linear,
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        ..Default::default()
    });

    for (entity, volumetric_fog_texture) in &views {
        let upsampling_bind_group = render_device.create_bind_group(
            "bloom_upsampling_bind_group",
            &upsampling_pipeline.bind_group_layout,
            &BindGroupEntries::sequential((
                &volumetric_fog_texture.view(),
                &sampler,
                uniforms.binding().unwrap(),
            )),
        );

        commands.entity(entity).insert(UpsamplingBindGroup {
            upsampling_bind_group,
            sampler: sampler.clone(),
        });
    }
}

#[derive(Component)]
pub struct UpsamplingPipelineIds {
    pub id_final: CachedRenderPipelineId,
}

#[derive(Resource)]
pub struct UpsamplingPipeline {
    pub bind_group_layout: BindGroupLayout,
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub struct FogUpsamplingPipelineKeys {}

impl FromWorld for UpsamplingPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let bind_group_layout = render_device.create_bind_group_layout(
            "fog_upsampling_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    // Input texture
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    // Sampler
                    sampler(SamplerBindingType::Filtering),
                    // FogUniforms
                    uniform_buffer::<UpsamplingUniforms>(true),
                ),
            ),
        );

        UpsamplingPipeline { bind_group_layout }
    }
}

pub const FOG_UPSCALING_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(0x14b0e0d8dbeb82cf729f6cc293554932);

impl SpecializedRenderPipeline for UpsamplingPipeline {
    type Key = FogUpsamplingPipelineKeys;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("fog_upsampling_pipeline".into()),
            layout: vec![self.bind_group_layout.clone()],
            vertex: fullscreen_shader_vertex_state(),
            fragment: Some(FragmentState {
                shader: FOG_UPSCALING_SHADER_HANDLE,
                shader_defs: vec![],
                entry_point: "upsample".into(),
                targets: vec![Some(ColorTargetState {
                    format: FOG_TEXTURE_FORMAT,
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::Constant,
                            dst_factor: BlendFactor::One,
                            operation: BlendOperation::Add,
                        },
                        alpha: BlendComponent {
                            src_factor: BlendFactor::Zero,
                            dst_factor: BlendFactor::One,
                            operation: BlendOperation::Add,
                        },
                    }),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            push_constant_ranges: Vec::new(),
            zero_initialize_workgroup_memory: false,
        }
    }
}

pub fn prepare_upsampling_pipeline(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<UpsamplingPipeline>>,
    pipeline: Res<UpsamplingPipeline>,
    views: Query<(Entity, &VolumetricFog)>,
) {
    for (entity, fog) in &views {
        let pipeline_final_id =
            pipelines.specialize(&pipeline_cache, &pipeline, FogUpsamplingPipelineKeys {});

        commands.entity(entity).insert(UpsamplingPipelineIds {
            id_final: pipeline_final_id,
        });
    }
}

impl ExtractComponent for VolumetricFog {
    type QueryData = (&'static Self, &'static Camera);

    type QueryFilter = ();
    type Out = (Self, UpsamplingUniforms);

    fn extract_component(
        (volumetric_fog, camera): QueryItem<'_, Self::QueryData>,
    ) -> Option<Self::Out> {
        match (
            camera.physical_viewport_rect(),
            camera.physical_viewport_size(),
            camera.physical_target_size(),
            camera.is_active,
            camera.hdr,
        ) {
            (Some(URect { min: origin, .. }), Some(size), Some(target_size), true, true)
                if size.x != 0 && size.y != 0 =>
            {
                let uniform = UpsamplingUniforms {
                    aspect: AspectRatio::try_from_pixels(size.x, size.y)
                        .expect("Valid screen size values for Bloom settings")
                        .ratio(),
                };

                Some((volumetric_fog.clone(), uniform))
            }
            _ => None,
        }
    }
}
