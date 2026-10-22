// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ELAST_ISOANISOEXPOACTIVE_HPP
#define FOUR_C_MAT_ELAST_ISOANISOEXPOACTIVE_HPP

#include "4C_config.hpp"

#include "4C_mat_anisotropy_extension_default.hpp"
#include "4C_mat_anisotropy_extension_provider.hpp"
#include "4C_mat_elast_activesummand.hpp"
#include "4C_mat_par_aniso.hpp"
#include "4C_material_parameter_base.hpp"

#include <memory>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace Elastic
  {
    namespace PAR
    {
      /*! 
       * @brief material parameters for an anisotropic active exponential fiber material
       *
       * <h3>Input line</h3>
       */
      class IsoAnisoExpoActive : public Mat::PAR::ParameterAniso
      {
       public:
        explicit IsoAnisoExpoActive(const Core::Mat::PAR::Parameter::Data& matdata);

        double gamma_;
        int init_;
        bool adapt_angle_;
        /// actin-myosin filament stiffness [MPa]
        double mue_smc_;
        /// resting state normalized relative sliding (= 0.1)
        double urs0_;
        /// contraction amplitude a0 (0.5 circular, 0.4 longitudinal)
        double a0_contraction_;
        /// Override this method and throw error, as the material should be created in the Factory method of the elastic summand
        /// speed of the activation wave (0 = constant activation everywhere)
        double v_;
        /// length of the contracting segment in reference config (= L in a(z,t))
        double delta_l_;
        /// time delay of activation initiation
        double t0_;
        std::shared_ptr<Core::Mat::Material> create_material() override
        {
          FOUR_C_THROW(
              "Cannot create a material from this method, as it should be created in "
              "Mat::Elastic::Summand::Factory.");
          return nullptr;
        };
      };  // class IsoAnisoExpoActive
    }    // namespace PAR

    /*! 
     * @brief Active isochoric contribution of an anisotropic exponential fiber material
     *
     * This material uses an active stress approach combined with an anisotropic exponential
     * fiber energy. The passive contribution is equivalent to IsoAnisoExpo, while the active
     * response is computed from a simple fiber activation potential.
     */
    class IsoAnisoExpoActive : public ActiveSummand, public FiberAnisotropyExtensionProvider<1>
    {
     public:
      explicit IsoAnisoExpoActive(Mat::Elastic::PAR::IsoAnisoExpoActive* params);

      void pack_summand(Core::Communication::PackBuffer& data) const override;

      void unpack_summand(Core::Communication::UnpackBuffer& buffer) override;

      Core::Materials::MaterialType material_type() const override
      {
        return Core::Materials::mes_isoanisoexpoactive;
      }

      void setup(int numgp, const Discret::Elements::Fibers& fibers,
          const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override;

      void register_anisotropy_extensions(Mat::Anisotropy& anisotropy) override;

      void evaluate_first_derivatives_aniso(Core::LinAlg::Matrix<2, 1>& dPI_aniso,
          const Core::LinAlg::SymmetricTensor<double, 3, 3>& rcg, int gp, int eleGID) override;

      void evaluate_second_derivatives_aniso(Core::LinAlg::Matrix<3, 1>& ddPII_aniso,
          const Core::LinAlg::SymmetricTensor<double, 3, 3>& rcg, int gp, int eleGID) override;

      template <typename T>
      void get_derivatives_aniso(Core::LinAlg::Matrix<2, 1, T>& dPI_aniso,
          Core::LinAlg::Matrix<3, 1, T>& ddPII_aniso,
          Core::LinAlg::Matrix<4, 1, T>& dddPIII_aniso,
          const Core::LinAlg::SymmetricTensor<T, 3, 3>& rcg, int gp, int eleGID) const;

      void add_stress_aniso_principal(
          const Core::LinAlg::SymmetricTensor<double, 3, 3>& rcg,
          Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat,
          Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
          const Teuchos::ParameterList& params, int gp, int eleGID) override;

      template <typename T>
      void evaluate_func(T& psi, Core::LinAlg::SymmetricTensor<T, 3, 3> const& rcg, int gp,
          int eleGID) const;

      void add_active_stress_cmat_aniso(const Core::LinAlg::SymmetricTensor<double, 3, 3>& CM,
          Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat,
          Core::LinAlg::SymmetricTensor<double, 3, 3>& stress, int gp, int eleGID,
          double totaltime) const override;

      template <typename T>
      void evaluate_active_stress_cmat_aniso(
          Core::LinAlg::SymmetricTensor<T, 3, 3> const& CM,
          Core::LinAlg::SymmetricTensor<T, 3, 3, 3, 3>& cmat,
          Core::LinAlg::SymmetricTensor<T, 3, 3>& stress, int gp, int eleGID) const;

      void add_strain_energy(double& psi, const Core::LinAlg::Matrix<3, 1>& prinv,
          const Core::LinAlg::Matrix<3, 1>& modinv,
          const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain, int gp,
          int eleGID) override;

      template <typename T>
      inline void get_derivative_aniso_active(T& dPIact) const
      {
        dPIact = d_p_iact_;
      };

      double get_derivative_aniso_active() const override { return d_p_iact_; };

      void set_fiber_vecs(double newgamma, const Core::LinAlg::Tensor<double, 3, 3>& locsys,
          const Core::LinAlg::Tensor<double, 3, 3>& defgrd) override;

      void get_fiber_vecs(std::vector<Core::LinAlg::Tensor<double, 3>>& fibervecs) const override;

      void specify_formulation(bool& isoprinc, bool& isomod, bool& anisoprinc,
          bool& anisomod, bool& viscogeneral) override
      {
        anisoprinc = true;
      };

      FiberAnisotropyExtension<1>& get_fiber_anisotropy_extension() override
      {
        return anisotropy_extension_;
      }

     private:
      double evaluated_psi_active() const;

      Mat::Elastic::PAR::IsoAnisoExpoActive* params_;
      double d_p_iact_;
      double lambdaact_;
      double z_ref_;
      DefaultAnisotropyExtension<1> anisotropy_extension_;
    };
  }  // namespace Elastic
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
